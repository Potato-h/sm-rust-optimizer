use super::*;

impl DataGraph {
    // FIXME: need to somehow recognize and drop unused stack variables.
    pub fn compile(&self, mut free_loc: u16) -> Vec<LinInst> {
        fn compile_node(
            node: NodeIndex,
            dag: &StableGraph<DataVertex, ArgStackOffset>,
            symbolics: &BTreeMap<Sym, NodeIndex>,
            already_compiled: &mut BTreeMap<NodeIndex, u16>,
            free_loc: &mut u16,
            code: &mut Vec<LinInst>,
        ) {
            eprintln!("compile: {:?}", dag[node]);

            if let Some(&loc) = already_compiled.get(&node) {
                code.push(LinInst::Load(Sym::Loc(loc)));
                return;
            }

            for dep in dag
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|e| e.weight())
                .rev()
            {
                compile_node(
                    dep.source(),
                    dag,
                    symbolics,
                    already_compiled,
                    free_loc,
                    code,
                );
            }

            match &dag[node] {
                DataVertex::Symbolic(sym) => {
                    code.push(LinInst::Load(sym.clone()));
                }
                DataVertex::StackVar(_) => {
                    // NOTE: should be careful here, invariants of analyzed stack machine code
                    // says that stack variable can only appear with "natural" computations, i. e.
                    // doesn't need to shift anything, but can add assertion
                }
                DataVertex::OpResult(inst) => {
                    code.push(inst.clone());
                }
            }

            let used_multiple_times = dag.edges_directed(node, Direction::Outgoing).count() > 1;
            let was_saved = already_compiled.contains_key(&node);
            let used_for_store_output_sym = symbolics
                .iter()
                .filter(|&(sym, node)| !matches!(&dag[*node], DataVertex::Symbolic(input) if input == sym))
                .any(|(_, n)| *n == node);

            if (used_multiple_times || used_for_store_output_sym) && !was_saved {
                let save = *free_loc;
                code.push(LinInst::Store(Sym::Loc(save)));
                eprintln!("save loc[{save}] for {:?}", dag[node]);
                *free_loc += 1;
                already_compiled.insert(node, save);
            }
        }

        eprintln!("start compile linear block: {}", self.start_label);

        let output_nodes: BTreeSet<_> = self
            .outputs
            .stack
            .iter()
            .cloned()
            .chain(self.jump_decided_by)
            .collect();

        eprintln!(
            "output nodes: {:?}",
            self.outputs
                .stack
                .iter()
                .cloned()
                .chain(self.jump_decided_by)
                .collect_vec()
        );

        let mut code = Vec::new();
        let mut already_compiled = BTreeMap::new();

        // Dirty hack: load all stack input in local variables to avoid
        // problems with extracting right stack variable at the end of
        // `compile_node` recursion.
        for (input, _) in self
            .inputs
            .iter()
            .filter_map(|&v| match self.dag[v] {
                DataVertex::StackVar(offset) => Some((v, offset)),
                _ => None,
            })
            .sorted_by_key(|(_, offset)| *offset)
        {
            code.push(LinInst::Store(Sym::Loc(free_loc)));
            code.push(LinInst::Drop);
            already_compiled.insert(input, free_loc);
            free_loc += 1;
        }

        // FIXME: need to compile dead nodes early to drop unused stack variables, but
        // this can backfire if unused stack input variable in between used.
        let dead_nodes = self
            .dag
            .node_indices()
            .filter(|v| !output_nodes.contains(v))
            .filter(|&v| self.dag.edges_directed(v, Direction::Outgoing).count() == 0);

        for node in dead_nodes {
            eprintln!("compile dead node: {:?}", self.dag[node]);

            compile_node(
                node,
                &self.dag,
                &self.symbolics,
                &mut already_compiled,
                &mut free_loc,
                &mut code,
            );

            // TODO: Do we need to always do this?
            code.push(LinInst::Drop);
        }

        for node in self
            .outputs
            .stack
            .iter()
            .cloned()
            .chain(self.jump_decided_by)
        {
            eprintln!("compile stack output node: {:?}", self.dag[node]);

            compile_node(
                node,
                &self.dag,
                &self.symbolics,
                &mut already_compiled,
                &mut free_loc,
                &mut code,
            );
        }

        // Store all needed variables only after computations within expression
        // to avoid mess up intermediate references to variables.
        for (sym, node) in self.symbolics.iter().filter(
            |&(sym, node)| !matches!(&self.dag[*node], DataVertex::Symbolic(input) if input == sym),
        ) {
            eprintln!("Try to find local tmp value for {sym} in {already_compiled:?}");
            let saved = already_compiled[node];
            code.extend([
                LinInst::Load(Sym::Loc(saved)),
                LinInst::Store(sym.clone()),
                LinInst::Drop,
            ]);
        }

        code
    }
}

impl FlowGraph {
    pub fn compile(&self, name: &str) -> (Vec<Inst>, u16, u16, u16) {
        fn compile_vertex(
            flow: &FlowGraph,
            node: NodeIndex,
            code: &mut Vec<Inst>,
            free_loc: u16,
            provide_label: Option<LabelMode>,
        ) {
            if let Some(mode) = provide_label {
                code.push(Inst::Flow(FlowInst::Label(flow.get_label(node), mode)));
            }

            match &flow.graph[node] {
                FlowVertex::LinearBlock(block) => {
                    code.extend(block.compile(free_loc).into_iter().map(Inst::Linear));
                }
                FlowVertex::Call(call) => {
                    code.push(Inst::Flow(FlowInst::Call(call.name.clone(), call.args)));
                }
                FlowVertex::STI(_) => code.push(Inst::Flow(FlowInst::STI)),
                FlowVertex::STA(_) => code.push(Inst::Flow(FlowInst::STA)),
                FlowVertex::CallC(callc) => code.push(Inst::Flow(FlowInst::CallC(callc.args))),
            }
        }

        eprintln!("start compile function: {name}");

        fn dfs(
            flow: &FlowGraph,
            current: NodeIndex,
            previous: Option<NodeIndex>,
            code: &mut Vec<Inst>,
            exit_label: &str,
            already_compiled: &mut BTreeSet<NodeIndex>,
            free_loc: u16,
        ) {
            already_compiled.insert(current);

            let provide_label = flow
                .graph
                .edges_directed(current, Direction::Incoming)
                .filter(|v| !previous.iter().contains(&v.source())) // all except previously compiled node
                .count()
                > 0;

            // FIXME: it's wrong way to compute mode with tail calls
            let label_was_known_before = flow
                .graph
                .edges_directed(current, Direction::Incoming)
                .filter(|v| already_compiled.contains(&v.source()))
                .count()
                > 0;

            let previous_jump_on_current = previous
                .iter()
                .flat_map(|&p| flow.graph.edges_directed(p, Direction::Outgoing))
                .any(|e| e.target() == current);

            let provide_label = match (
                provide_label,
                label_was_known_before,
                previous_jump_on_current,
            ) {
                (true, true, _) => Some(LabelMode::RetrieveStack),
                (true, _, false) => Some(LabelMode::RetrieveStack),
                (true, false, _) => Some(LabelMode::DropBarrier),
                _ => None,
            };

            compile_vertex(flow, current, code, free_loc, provide_label);

            if let Some(outs) = flow.jmp_outgoings(current) {
                if let Some((mode, to)) = outs.cjmp {
                    code.push(Inst::Flow(FlowInst::Jmp(mode, flow.get_label(to))));
                }

                if !already_compiled.contains(&outs.jmp) {
                    dfs(
                        flow,
                        outs.jmp,
                        Some(current),
                        code,
                        exit_label,
                        already_compiled,
                        free_loc,
                    );
                } else {
                    code.push(Inst::Flow(FlowInst::Jmp(
                        JumpMode::Unconditional,
                        flow.get_label(outs.jmp),
                    )));
                }
            }

            if flow.outputs.contains(&current) {
                code.push(Inst::Flow(FlowInst::Jmp(
                    JumpMode::Unconditional,
                    exit_label.to_string(),
                )));
            }
        }

        let exit_label = format!("{name}_exit");
        let mut code = Vec::new();
        let free_loc = self.loc_count();
        let mut already_compiled = BTreeSet::new();

        for node in iter::once(self.input).chain(self.graph.node_indices()) {
            if !already_compiled.contains(&node) {
                dfs(
                    self,
                    node,
                    None,
                    &mut code,
                    &exit_label,
                    &mut already_compiled,
                    free_loc,
                );
            }
        }

        let actual_locs = code
            .iter()
            .filter_map(|inst| match inst {
                Linear(LinInst::Store(Sym::Loc(l))) => Some(*l + 1),
                Linear(LinInst::Load(Sym::Loc(l))) => Some(*l + 1),
                Linear(LinInst::LDA(Sym::Loc(l))) => Some(*l + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        code.push(Inst::Flow(FlowInst::Label(
            exit_label,
            LabelMode::RetrieveStack,
        )));
        (code, self.args_count(), actual_locs, self.clos_count())
    }
}

impl Unit {
    pub fn compile(&self) -> Vec<Inst> {
        let mut code = self
            .declarations
            .iter()
            .cloned()
            .map(Inst::Decl)
            .collect_vec();

        for (name, function) in self
            .functions
            .iter()
            // FIXME: "main" function should go first in code generation
            // actually should find it in `self.declarations`
            .sorted_by_key(|(name, _)| !name.contains("init"))
        {
            let (body, args, locs, clos) = function.compile(name);
            // For functions LabelMode is useless
            code.push(Inst::Flow(FlowInst::Label(
                name.clone(),
                LabelMode::RetrieveStack,
            )));
            code.push(Inst::Flow(FlowInst::Begin(Begin {
                name: name.clone(),
                args: args as u32,
                locs: locs as u32,
                clos: clos as u32,
            })));
            code.extend(body);
            code.push(Inst::Flow(FlowInst::End));
        }

        code
    }
}
