use super::*;

impl DataGraph {
    pub fn extend(&mut self, ext: &DataGraph) {
        assert!(self.jump_decided_by.is_none());

        let ext_to_source = add_graph(&mut self.dag, &ext.dag);

        let ext_stack_vars_in_source = ext.inputs.iter().filter_map(|&node| match ext.dag[node] {
            DataVertex::StackVar(offset) => Some((ext_to_source[&node], offset)),
            _ => None,
        });

        let mut ext_deleted_inputs = BTreeMap::new();

        for (input, _) in ext_stack_vars_in_source.sorted_by_key(|(_, offset)| *offset) {
            let output = self.outputs.pop().right_or_else(|var| {
                let new_stack_arg = self.dag.add_node(var);
                self.inputs.push(new_stack_arg);
                new_stack_arg
            });

            let input_outgoings = self
                .dag
                .edges_directed(input, Direction::Outgoing)
                .map(|edge| (edge.target(), *edge.weight()))
                .collect_vec();

            for (to, w) in input_outgoings {
                self.dag.add_edge(output, to, w);
            }

            ext_deleted_inputs.insert(input, output);
            self.dag.remove_node(input);
        }

        // update ext_to_source after removing some input nodes in ext
        let ext_to_source = |node: NodeIndex| {
            *ext_deleted_inputs
                .get(&ext_to_source[&node])
                .unwrap_or(&ext_to_source[&node])
        };

        // Merge results of 2 stacks. Now self contains actual count of
        // virtual nodes, but ext contains actual output nodes that will be on stack
        // after block execution
        self.outputs
            .stack
            .extend(ext.outputs.stack.iter().map(|&node| ext_to_source(node)));

        let ext_sym_vars_in_source = ext.inputs.iter().filter_map(|&node| match &ext.dag[node] {
            DataVertex::Symbolic(name) => Some((ext_to_source(node), name)),
            _ => None,
        });

        for (sym, name) in ext_sym_vars_in_source {
            if let Some(sym_in_source) = self.symbolics.get(name) {
                // If source block has symbolic value, then just change node
                // description to store operation from last symbolic value in source
                self.dag[sym] = DataVertex::OpResult(LinInst::Store(name.clone()));
                self.dag.add_edge(*sym_in_source, sym, 0);
            } else {
                // Otherwise keep symbolic input node
                self.inputs.push(sym);
            }
        }

        // if symbolic value was never read in block, then there is no corresponding
        // symbolic input node, so need to update all symbolics
        for (name, sym_in_ext) in ext.symbolics.iter() {
            self.symbolics
                .insert(name.clone(), ext_to_source(*sym_in_ext));
        }

        self.jump_decided_by = ext.jump_decided_by.as_ref().map(|&id| ext_to_source(id));
        self.check_invariants();
    }

    pub fn remove_jump_decision(&mut self) {
        if let Some(jump) = self.jump_decided_by.take() {
            self.dag.remove_node(jump);
        }
    }

    pub fn remove_stores(&mut self) {
        while let Some(store) = self
            .dag
            .node_indices()
            .find(|&id| matches!(&self.dag[id], DataVertex::OpResult(LinInst::Store(_))))
        {
            eprintln!("Found store: {store:?}");

            let store_source = self
                .dag
                .edges_directed(store, Direction::Incoming)
                .map(|edge| edge.source())
                .collect_vec();

            let store_usages = self
                .dag
                .edges_directed(store, Direction::Outgoing)
                .map(|edge| (edge.target(), *edge.weight()))
                .collect_vec();

            // this loop must have exactly one iteration
            for (i, from) in store_source.into_iter().enumerate() {
                assert!(i == 0, "this loop must have exactly one iteration");

                for (to, w) in store_usages.iter().cloned() {
                    self.dag.add_edge(from, to, w);
                }

                self.change_output_state(store, from);
            }

            self.dag.remove_node(store);
        }

        self.check_invariants();
    }

    // NOTE: is removal of input node breaks relation of block? probably not,
    // because it's only matters for stack values, but we annotate stack depth
    // for each node.
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // this is actually false. for example we can have following structure
    //
    // A {0, 1} -> B {1} -> C {0}. if B contains dead code dependent on stack variables, then
    // stack variable by itself preserve stack levels used in C. If we remove all information
    // about using {1} in B, then after block merge, C will use {1} not desired {0}.
    //
    // TODO: is some cases symbolic variable may be useless for specific block,
    // but actually bypassed for next blocks
    pub fn eliminate_dead_code(&mut self) {
        let stack_inputs: BTreeSet<_> = self
            .inputs
            .iter()
            .cloned()
            .filter(|&input| matches!(self.dag[input], DataVertex::StackVar(_)))
            .collect();

        let outputs: BTreeSet<_> = self
            .jump_decided_by
            .iter()
            .chain(self.outputs_nodes().iter())
            .chain(self.symbolics.values())
            .cloned()
            .collect();

        while let Some(sink) = self
            .dag
            .node_indices()
            .filter(|id| !stack_inputs.contains(id))
            .filter(|id| !outputs.contains(id))
            .find(|id| self.dag.edges_directed(*id, Direction::Outgoing).count() == 0)
        {
            self.dag.remove_node(sink);
            self.inputs.retain(|&x| x != sink);
        }

        self.check_invariants();
    }

    pub fn change_output_state(&mut self, node: NodeIndex, new_node: NodeIndex) {
        self.outputs
            .stack
            .iter_mut()
            .filter(|out_node| **out_node == node)
            .for_each(|out_node| *out_node = new_node);

        if self.jump_decided_by.is_some_and(|jmp| jmp == node) {
            self.jump_decided_by = Some(new_node);
        }

        self.symbolics
            .values_mut()
            .filter(|s| **s == node)
            .for_each(|s| *s = new_node);

        self.check_invariants();
    }

    /// Replace node with new value and saves *only* outgoing edges from this node.
    /// All incoming edges deleted. Replaces all entries of this node in special nodes (stack outputs,
    /// symbolics, jump decision variable) with new node.
    pub fn replace_node_for_outgoings(
        &mut self,
        node: NodeIndex,
        new_node: DataVertex,
    ) -> NodeIndex {
        let new_node = self.dag.add_node(new_node);

        let outgoing = self
            .dag
            .edges_directed(node, Direction::Outgoing)
            .map(|edge| (edge.target(), *edge.weight()))
            .collect_vec();

        for (to, arg_pos) in outgoing {
            self.dag.add_edge(new_node, to, arg_pos);
        }

        self.change_output_state(node, new_node);
        self.dag.remove_node(node);
        self.check_invariants();
        new_node
    }

    pub fn constant_propagation(&mut self) {
        while let Some((node, op)) = self
            .dag
            .node_references()
            .filter_map(|(i, v)| match v {
                DataVertex::OpResult(LinInst::BinOp(op)) => Some((i, op)),
                _ => None,
            })
            .find(|(i, _)| {
                self.dag
                    .neighbors_directed(*i, Direction::Incoming)
                    .all(|v| matches!(self.dag[v], DataVertex::OpResult(LinInst::Const(_))))
            })
        {
            let Some([lhs, rhs]) = self
                .dag
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|edge| edge.weight())
                .rev()
                .filter_map(|edge| match self.dag[edge.source()] {
                    DataVertex::OpResult(LinInst::Const(v)) => Some(v),
                    _ => None,
                })
                .collect_array()
            else {
                break;
            };

            let value = op.eval(lhs, rhs);
            self.replace_node_for_outgoings(node, DataVertex::OpResult(LinInst::Const(value)));
        }

        self.check_invariants();
    }

    pub fn tag_check_evaluation(&mut self) {
        while let Some((node, new_value)) = self
            .dag
            .edge_references()
            .filter_map(
                |edge| match (&self.dag[edge.source()], &self.dag[edge.target()]) {
                    (
                        DataVertex::OpResult(LinInst::SExp(have_tag, have_args)),
                        DataVertex::OpResult(LinInst::Pat(Pat::Tag(expect_tag, expect_args))),
                    ) => {
                        if have_tag == expect_tag && have_args == expect_args {
                            Some((edge.target(), 1))
                        } else {
                            Some((edge.target(), 0))
                        }
                    }
                    _ => None,
                },
            )
            .next()
        {
            self.replace_node_for_outgoings(node, DataVertex::OpResult(LinInst::Const(new_value)));
        }

        self.check_invariants();
    }

    pub fn optimize(&mut self, flags: DataGraphOptimFlags) {
        if flags.elim_stores {
            self.remove_stores();
        }

        if flags.elim_dead_code {
            self.eliminate_dead_code();
        }

        if flags.const_prop {
            self.constant_propagation();
            self.eliminate_dead_code();
        }

        if flags.tag_eval {
            self.tag_check_evaluation();
            self.eliminate_dead_code();
        }
    }

    pub fn shift_symbolics_for_inline(&mut self, first_free_loc: u16, args_count: u16) {
        for node in self.dag.node_weights_mut() {
            if let DataVertex::Symbolic(sym) = node {
                *sym = sym_shift(sym.clone(), first_free_loc, args_count);
            }
        }

        self.symbolics = mem::take(&mut self.symbolics)
            .into_iter()
            .map(|(k, v)| (sym_shift(k, first_free_loc, args_count), v))
            .collect();
    }

    pub fn remove_stores_to_unused_symbolics(&mut self, read_symbolics: &BTreeSet<Sym>) {
        let symbolics_to_delete = self
            .symbolics
            .keys()
            .filter(|k| !read_symbolics.contains(k))
            .cloned()
            .collect_vec();

        for sym in symbolics_to_delete {
            self.symbolics.remove(&sym);
        }
    }
}

fn sym_shift(sym: Sym, first_free_loc: u16, args_count: u16) -> Sym {
    match sym {
        Sym::Arg(i) => Sym::Loc(first_free_loc + i),
        Sym::Loc(i) => Sym::Loc(first_free_loc + args_count + i),
        Sym::Glb(glb) => Sym::Glb(glb),
        Sym::Acc(_) => todo!("Implement inlining of closure later"),
    }
}

impl FlowGraph {
    pub fn eliminate_dead_code(&mut self) {
        while let Some(source) = self
            .graph
            .node_indices()
            .filter(|&id| id != self.input)
            .find(|&id| self.graph.edges_directed(id, Direction::Incoming).count() == 0)
        {
            self.graph.remove_node(source);
        }
    }

    pub fn jump_on_const(&mut self) {
        while let Some((node, v)) = self
            .graph
            .node_references()
            .filter_map(|(id, v)| v.block().map(|block| (id, block)))
            .filter(|(id, _)| {
                self.graph
                    .edges_directed(*id, Direction::Outgoing)
                    .map(|edge| edge.weight())
                    .any(JumpCondition::is_conditional)
            })
            .find_map(|(id, block)| {
                block.jump_decided_by.and_then(|jmp| match block.dag[jmp] {
                    DataVertex::OpResult(LinInst::Const(v)) => Some((id, v)),
                    _ => None,
                })
            })
        {
            if let FlowVertex::LinearBlock(block) = &mut self.graph[node] {
                block.remove_jump_decision();
            }

            let outgoings = self
                .graph
                .edges_directed(node, Direction::Outgoing)
                .map(|edge| (edge.id(), *edge.weight()))
                .collect_vec();

            for (edge, cond) in outgoings {
                if cond.will_jump(v) {
                    self.graph[edge] = JumpCondition::Unconditional;
                } else {
                    self.graph.remove_edge(edge);
                }
            }
        }
    }

    pub fn merge_block_with_unconditional_jump(&mut self) {
        while let Some((from, to)) = self
            .graph
            .node_references()
            .filter(|(id, vertex)| *id != self.input && vertex.is_block())
            .find_map(|(to, _)| {
                self.graph
                    .edges_directed(to, Direction::Incoming)
                    .map(|edge| (edge.source(), *edge.weight()))
                    .collect_array()
                    .and_then(|x| match x {
                        [(from, JumpCondition::Unconditional)] if self.graph[from].is_block() => {
                            Some((from, to))
                        }
                        _ => None,
                    })
            })
        {
            eprintln!(
                "merging blocks: {} -> {}",
                self.get_label(from),
                self.get_label(to)
            );

            let ext = if let FlowVertex::LinearBlock(block) = &self.graph[to] {
                block.clone()
            } else {
                unreachable!("by filter")
            };

            if let FlowVertex::LinearBlock(block) = &mut self.graph[from] {
                block.extend(&ext);
            }

            let outgoings = self
                .graph
                .edges_directed(to, Direction::Outgoing)
                .map(|edge| (edge.target(), *edge.weight()))
                .collect_vec();

            for (to, w) in outgoings {
                self.graph.add_edge(from, to, w);
            }

            // Since `from` and `to` is one block, we should preserve output status in merged block
            if self.outputs.contains(&to) {
                self.outputs.retain(|&out| out != from && out != to);
                self.outputs.push(from);
            }

            if let FlowVertex::LinearBlock(block) = &mut self.graph[from] {
                // Since stores break code generation, get rid of generated stores
                block.remove_stores();
            }

            self.graph.remove_node(to);
        }
    }

    pub fn optimize(&mut self, flow_optim: FlowOptimFlags) {
        for _ in 0..flow_optim.passes {
            for block in self.graph.node_weights_mut() {
                if let FlowVertex::LinearBlock(graph) = block {
                    graph.optimize(flow_optim.data_flags);
                }
            }

            if flow_optim.elim_dead_code {
                self.eliminate_dead_code();
            }

            if flow_optim.jump_on_const {
                self.jump_on_const();
                self.eliminate_dead_code();
            }

            if flow_optim.merge_blocks {
                self.merge_block_with_unconditional_jump();
            }

            if flow_optim.liveliness_analysis {
                self.remove_unused_symbolics();
            }
        }
    }

    pub fn replace_node_with_graph(&mut self, node: NodeIndex, replacement: &FlowGraph) {
        assert_ne!(self.input, node, "Can't replace input node with graph");

        for edge in self.graph.edges_directed(node, Direction::Outgoing) {
            assert_eq!(
                *edge.weight(),
                JumpCondition::Unconditional,
                "All jumps from replaced node should be unconditional"
            );
        }

        let incomings = self
            .graph
            .edges_directed(node, Direction::Incoming)
            .map(|edge| (edge.source(), *edge.weight()))
            .collect_vec();

        let outgoings = self
            .graph
            .edges_directed(node, Direction::Outgoing)
            .map(|edge| edge.target())
            .collect_vec();

        let replacement_remapping = add_graph(&mut self.graph, &replacement.graph);

        for (from, cond) in incomings {
            self.graph
                .add_edge(from, replacement_remapping[&replacement.input], cond);
        }

        for out in replacement.outputs.iter() {
            let out = replacement_remapping[out];

            for to in outgoings.iter() {
                self.graph.add_edge(out, *to, JumpCondition::Unconditional);
            }
        }

        if let Some((i, _)) = self.outputs.iter().find_position(|&&v| node == v) {
            self.outputs.swap_remove(i);
            self.outputs.extend(
                replacement
                    .outputs
                    .iter()
                    .map(|out| replacement_remapping[out]),
            );
        }

        self.graph.remove_node(node);
    }

    // TODO: how to actually deal with symbolics after inlining?
    // TODO: need to insert additional linear block that will deal
    //  with passing function arguments from stack
    pub fn replace_call_with_graph(
        &mut self,
        ctx: &mut Ctx,
        call: NodeIndex,
        mut replacement: FlowGraph,
    ) {
        let first_free_loc = self.loc_count();
        let call_args = replacement.args_count();
        replacement.shift_symbolics_for_inline(first_free_loc);
        replacement.shift_labels_for_inline(ctx);

        let prev_input = replacement.input;
        let prologue = gen_inline_call_prologue(ctx.fresh_label(), call_args, first_free_loc);
        let input = replacement
            .graph
            .add_node(FlowVertex::LinearBlock(prologue));

        replacement.input = input;
        replacement
            .graph
            .add_edge(input, prev_input, JumpCondition::Unconditional);

        self.replace_node_with_graph(call, &replacement);
    }

    pub fn replace_all_calls(
        &mut self,
        ctx: &mut Ctx,
        call: &str,
        replacement: &FlowGraph,
        mut unfold_limit: u32,
    ) {
        let expect_args = replacement.args_count();

        while let Some(node) = self.graph.node_references().find_map(|(id, v)| match v {
            FlowVertex::Call(call_v) if call_v.name == call => {
                assert_eq!(
                    call_v.args as u16, expect_args,
                    "while inlining {call}, number of arguments mismatch"
                );
                Some(id)
            }
            _ => None,
        }) {
            if unfold_limit == 0 {
                break;
            }

            self.replace_call_with_graph(ctx, node, replacement.clone());
            unfold_limit -= 1;
        }
    }

    fn shift_symbolics_for_inline(&mut self, first_free_loc: u16) {
        let args = self.args_count();

        for node in self.graph.node_weights_mut() {
            if let FlowVertex::LinearBlock(block) = node {
                block.shift_symbolics_for_inline(first_free_loc, args);
            }
        }
    }

    fn shift_labels_for_inline(&mut self, ctx: &mut Ctx) {
        for node in self.graph.node_weights_mut() {
            let label = ctx.fresh_label();

            match node {
                FlowVertex::LinearBlock(block) => block.start_label = label,
                FlowVertex::Call(call) => call.label = label,
                FlowVertex::STI(sti) => sti.label = label,
                FlowVertex::STA(sta) => sta.label = label,
                FlowVertex::CallC(call_c) => call_c.label = label,
            }
        }
    }

    // TODO: is this `nodes` allocation really needed?
    pub fn remove_unused_symbolics(&mut self) {
        let read_symbolics = self.find_read_symbolics();
        let nodes = self.graph.node_indices().collect_vec();

        for node in nodes {
            if let FlowVertex::LinearBlock(block) = &mut self.graph[node] {
                block.remove_stores_to_unused_symbolics(&read_symbolics[&node]);
            }
        }
    }

    pub fn strip_empty_output_nodes(&mut self) {
        while let Some((i, node)) = self
            .outputs
            .iter()
            .cloned()
            .enumerate()
            .filter(|&(_, v)| matches!(&self.graph[v], FlowVertex::LinearBlock(block) if block.dag.node_count() == 0))
            .find(|&(_, v)| self.graph.edges_directed(v, Direction::Incoming).all(|e| !e.weight().is_conditional()))
        {
            self.outputs.swap_remove(i);

            for incoming in self.graph.neighbors_directed(node, Direction::Incoming) {
                self.outputs.push(incoming);
            }

            self.graph.remove_node(node);
        }
    }

    pub fn replace_tail_call(&mut self, call: &str) {
        while let Some((i, node, label, args)) =
            self.outputs
                .iter()
                .cloned()
                .enumerate()
                .find_map(|(i, v)| match &self.graph[v] {
                    FlowVertex::Call(call_v) if call_v.name == call => {
                        Some((i, v, call_v.label.clone(), call_v.args))
                    }
                    _ => None,
                })
        {
            let replacement = gen_tail_call_prologue(label, args as u16);
            *(&mut self.graph[node]) = FlowVertex::LinearBlock(replacement);
            self.outputs.swap_remove(i);
            self.graph
                .add_edge(node, self.input, JumpCondition::Unconditional);
        }
    }
}

impl Unit {
    pub fn optimize(&mut self, ctx: &mut Ctx, flags: UnitOptimFlags) {
        for _ in 0..flags.passes {
            self.functions
                .values_mut()
                .for_each(|flow| flow.optimize(flags.flow_optim));

            let candidates = if flags.inline_strategy {
                self.find_candidates_for_inlining()
            } else {
                Vec::new()
            };

            for call in flags
                .force_inline
                .iter()
                .chain(candidates.into_iter().as_ref())
            {
                let call_graph = self.functions[call].clone();

                self.functions.values_mut().for_each(|flow| {
                    flow.replace_all_calls(ctx, call, &call_graph, 1);
                    flow.optimize(flags.flow_optim);
                });
            }

            // TODO: move to FlowGraph::optimize?
            if flags.flow_optim.tail_call {
                for (name, function) in self.functions.iter_mut() {
                    function.strip_empty_output_nodes();
                    function.replace_tail_call(&name);
                }
            }

            if flags.remove_unused_decls {
                self.remove_unused_decls();
            }
        }
    }

    pub fn find_candidates_for_inlining(&self) -> Vec<Ident> {
        self.functions
            .iter()
            .filter(|(_, function)| !function.has_calls() && function.graph.node_count() < 10)
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn find_used_decls(&self) -> BTreeSet<Ident> {
        let mut queue: VecDeque<_> = self
            .declarations
            .iter()
            .filter_map(|decl| match decl {
                common::Decl::Public(Public::Fun(name, _, _)) => Some(name.clone()),
                _ => None,
            })
            .chain(
                self.functions
                    .keys()
                    .filter(|name| name.contains("init"))
                    .cloned(),
            )
            .collect();

        let mut used_decls: BTreeSet<_> = queue.iter().cloned().collect();

        while let Some(decl) = queue.pop_front() {
            if let Some(flow) = self.functions.get(&decl) {
                let mut additional_decls = flow.find_used_decls();
                for decl in additional_decls.iter() {
                    if !used_decls.contains(decl) {
                        queue.push_back(decl.clone());
                    }
                }

                used_decls.append(&mut additional_decls);
            }
        }

        used_decls
    }

    pub fn remove_unused_decls(&mut self) {
        let used_decls = self.find_used_decls();

        self.functions = mem::take(&mut self.functions)
            .into_iter()
            .filter(|(k, _)| used_decls.contains(k))
            .collect();

        eprintln!("actually used decls: {used_decls:?}");
    }
}
