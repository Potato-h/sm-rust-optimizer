use super::*;

fn inst_vertex(inst: &LinInst) -> DataVertex {
    DataVertex::OpResult(inst.clone())
}

impl DataGraph {
    pub fn analyze(start_label: String, code: Vec<LinInst>, has_cjmp: bool) -> DataGraph {
        let mut dag = StableGraph::new();
        let mut stack = VirtualStack::new();
        let mut symbolics: BTreeMap<Sym, NodeIndex> = BTreeMap::new();

        fn n_arity_inst(
            dag: &mut StableGraph<DataVertex, usize>,
            stack: &mut VirtualStack,
            n: usize,
            inst: LinInst,
        ) {
            let node = dag.add_node(inst_vertex(&inst));

            for i in 0..n {
                let arg = stack.pop().right_or_else(|var| dag.add_node(var));
                dag.add_edge(arg, node, i);
            }

            stack.push(node);
        }

        eprintln!("code: {code:?}");

        for inst in code {
            eprintln!("stack: {stack:?}");

            // Assume, that all of the instructions consume stack values, aside from ST
            match inst {
                LinInst::Const(_) => n_arity_inst(&mut dag, &mut stack, 0, inst),
                LinInst::String(_) => n_arity_inst(&mut dag, &mut stack, 0, inst),
                LinInst::Elem => n_arity_inst(&mut dag, &mut stack, 2, inst),
                LinInst::Store(ref sym) => {
                    let node = dag.add_node(inst_vertex(&inst));
                    let from = match stack.peek() {
                        Left(v) => {
                            let node = dag.add_node(v);
                            stack.pop(); // move lazy input stack variable
                            stack.push(node); // reify lazy input stack variable
                            node
                        }
                        Right(node) => node,
                    };
                    dag.add_edge(from, node, 0);
                    symbolics.insert(sym.clone(), node);
                }
                LinInst::Load(ref sym) => {
                    if !symbolics.contains_key(sym) {
                        let node = dag.add_node(DataVertex::Symbolic(sym.clone()));
                        symbolics.insert(sym.clone(), node);
                    }

                    stack.push(symbolics[sym]);
                }
                LinInst::LDA(_) => n_arity_inst(&mut dag, &mut stack, 0, inst),
                LinInst::BinOp(_) => n_arity_inst(&mut dag, &mut stack, 2, inst),
                LinInst::Pat(_) => n_arity_inst(&mut dag, &mut stack, 1, inst),
                LinInst::SExp(_, n) => n_arity_inst(&mut dag, &mut stack, n, inst),
                LinInst::Closure(_, n) => n_arity_inst(&mut dag, &mut stack, n, inst),
                LinInst::Array(n) => n_arity_inst(&mut dag, &mut stack, n, inst),
                LinInst::Dup => {
                    match stack.peek() {
                        Left(stack_input_var) => {
                            let index = dag.add_node(stack_input_var);
                            stack.pop(); // move lazy input to next variable
                            stack.push(index); // reify input stack variable
                            stack.push(index); // actually perform DUP operation
                        }
                        Right(index) => {
                            stack.push(index);
                        }
                    }
                }
                LinInst::Drop => {
                    // TODO: actually, is this right? we can do drop at the start of linear block,
                    // but silent `pop` will just ignore dependency from blocks into that block
                    stack.pop().right_or_else(|var| dag.add_node(var));
                }
            }
        }

        let jump_decided_by = if has_cjmp {
            Some(stack.pop().right_or_else(|var| dag.add_node(var)))
        } else {
            None
        };

        let inputs = dag
            .node_references()
            .filter_map(|(i, typ)| match typ {
                DataVertex::StackVar(_) => Some(i),
                DataVertex::Symbolic(_) => Some(i),
                DataVertex::OpResult(_) => None,
            })
            .collect();

        assert!(
            !algo::is_cyclic_directed(&dag),
            "found cycle in data flow graph"
        );

        let mut graph = DataGraph {
            start_label,
            dag,
            inputs,
            outputs: stack,
            symbolics,
            jump_decided_by,
        };

        graph.remove_stores();
        graph
    }
}

impl FlowGraph {
    pub fn analyze(ctx: &mut Ctx, code: Vec<Inst>) -> Self {
        let mut edges: Vec<(NodeIndex, Label, JumpCondition)> = Vec::new();
        let mut edge_from_prev: Option<(NodeIndex, JumpCondition)> = None;
        let mut block: Vec<LinInst> = Vec::new();
        let mut graph = StableGraph::new();
        let mut labeled: Option<Label> = None;
        let mut block_indexes = Vec::new(); // only purpose is to find input and outputs blocks
        let mut from_label_to_block: BTreeMap<Label, NodeIndex> = BTreeMap::new();

        // skip(1) for Begin operation
        for inst in code.into_iter().skip(1) {
            match inst {
                Inst::Linear(inst) => {
                    block.push(inst);
                }
                Inst::Flow(inst) => {
                    let start_label = labeled.take().unwrap_or(ctx.fresh_label());
                    let has_cjmp = inst.conditional_jmp();
                    let this_block =
                        DataGraph::analyze(start_label.clone(), mem::take(&mut block), has_cjmp);

                    let node = graph.add_node(FlowVertex::LinearBlock(this_block));
                    from_label_to_block.insert(start_label, node);
                    block_indexes.push(node);

                    if let Some((prev, cond)) = edge_from_prev.take() {
                        graph.add_edge(prev, node, cond);
                    }

                    match inst {
                        FlowInst::Jmp(JumpMode::Unconditional, label) => {
                            edges.push((node, label, JumpCondition::Unconditional));
                        }
                        FlowInst::Jmp(jump_mode, label) => {
                            edges.push((node, label, jump_mode.into()));
                            edge_from_prev = Some((node, jump_mode.rev().into()))
                        }
                        FlowInst::Label(label, _) => {
                            edge_from_prev = Some((node, JumpCondition::Unconditional));
                            labeled = Some(label);
                        }
                        FlowInst::Call(name, args) => {
                            let call_node = graph.add_node(FlowVertex::Call(CallVertex {
                                label: ctx.fresh_label(),
                                name,
                                args,
                            }));
                            graph.add_edge(node, call_node, JumpCondition::Unconditional);
                            edge_from_prev = Some((call_node, JumpCondition::Unconditional));
                        }
                        FlowInst::CallC(args) => {
                            let call_node = graph.add_node(FlowVertex::CallC(CallCVertex {
                                label: ctx.fresh_label(),
                                args,
                            }));
                            graph.add_edge(node, call_node, JumpCondition::Unconditional);
                            edge_from_prev = Some((call_node, JumpCondition::Unconditional));
                        }
                        FlowInst::STI => {
                            let sti_node = graph.add_node(FlowVertex::STI(STIVertex {
                                label: ctx.fresh_label(),
                            }));
                            graph.add_edge(node, sti_node, JumpCondition::Unconditional);
                            edge_from_prev = Some((sti_node, JumpCondition::Unconditional));
                        }
                        FlowInst::STA => {
                            let sta_node = graph.add_node(FlowVertex::STA(STAVertex {
                                label: ctx.fresh_label(),
                            }));
                            graph.add_edge(node, sta_node, JumpCondition::Unconditional);
                            edge_from_prev = Some((sta_node, JumpCondition::Unconditional));
                        }
                        FlowInst::Begin(_) => continue,
                        FlowInst::End => break,
                    }
                }
                Inst::Decl(_) => {}
            }
        }

        for (from, to, cond) in edges {
            graph.add_edge(from, from_label_to_block[&to], cond);
        }

        FlowGraph {
            graph,
            input: block_indexes[0],
            outputs: block_indexes.last().into_iter().cloned().collect(),
        }
    }
}

impl Unit {
    pub fn analyze(ctx: &mut Ctx, code: impl Iterator<Item = Inst>) -> Self {
        let mut functions: BTreeMap<Ident, FlowGraph> = BTreeMap::new();
        let mut declarations = Vec::new();
        let mut code = code
            .filter(|inst| match inst {
                Inst::Decl(decl) => {
                    declarations.push(decl.clone());
                    false
                }
                _ => true,
            })
            .peekable();

        while let Some(Flow(FlowInst::Label(name, _))) = code.peek() {
            let name = name.clone();
            let code: Vec<_> = (&mut code)
                .skip(1)
                .take_while_inclusive(|inst| inst != &Flow(FlowInst::End))
                .collect();

            functions.insert(name, FlowGraph::analyze(ctx, code));
        }

        Unit {
            functions,
            declarations,
        }
    }
}
