use std::collections::BTreeMap;
use std::fmt::{self, Write};

use daggy::{Dag, NodeIndex};
use either::Either::{self, Left, Right};
use itertools::Itertools;
use petgraph::dot::Config::GraphContentOnly;
use petgraph::visit::{EdgeRef, IntoNodeReferences, NodeRef};
use petgraph::{dot::Dot, Graph};

type Ident = String;

#[derive(Debug, Clone, Copy)]
enum JumpMode {
    Unconditional,
    Zero,
    NonZero,
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Plus,
    Minus,
}

#[derive(Debug, Clone)]
enum LinInst {
    Const(i32),
    Elem,
    Store(Ident),
    Load(Ident),
    Call(Ident, usize),
    BinOp(Op),
    Tag(Ident, usize),
    SExp(Ident, usize),
    Dup,
    Drop,
}

#[derive(Debug, Clone)]
enum FlowInst {
    Jmp(JumpMode, Ident),
    Label(Ident),
    Ret,
    Begin,
    End,
}

#[derive(Debug, Clone)]
enum Inst {
    LinInst(LinInst),
    FlowInst(FlowInst),
}

type ArgStackOffset = usize;

#[derive(Debug, Clone)]
struct DataGraph {
    dag: Dag<DataVertex, ArgStackOffset>,
    inputs: Vec<NodeIndex>,
    outputs: Vec<NodeIndex>,
}

#[derive(Debug, Clone, Copy)]
enum JumpCondition {
    Unconditional,
    Zero,
    NonZero,
}

#[derive(Debug, Clone)]
enum DataVertex {
    StackVar(usize),
    OpResult(LinInst),
}

struct VirtualStack {
    tail_from: usize,
    stack: Vec<NodeIndex>,
}

impl VirtualStack {
    fn new() -> Self {
        VirtualStack {
            tail_from: 0,
            stack: Vec::new(),
        }
    }

    fn push(&mut self, v: NodeIndex) {
        self.stack.push(v);
    }

    fn pop(&mut self) -> Either<DataVertex, NodeIndex> {
        match self.stack.pop() {
            Some(v) => Right(v),
            None => {
                let tail = self.tail_from;
                self.tail_from += 1;
                Left(DataVertex::StackVar(tail))
            }
        }
    }

    fn peek(&self) -> Either<DataVertex, NodeIndex> {
        match self.stack.last() {
            Some(v) => Right(v.clone()),
            None => {
                let tail = self.tail_from;
                Left(DataVertex::StackVar(tail))
            }
        }
    }
}

type FlowGraph = Graph<DataGraph, JumpCondition>;

fn compile_lin_block(code: Vec<LinInst>) -> DataGraph {
    let mut dag = Dag::new();
    let mut stack = VirtualStack::new();

    fn n_arity_inst(
        dag: &mut Dag<DataVertex, usize>,
        stack: &mut VirtualStack,
        n: usize,
        inst: LinInst,
    ) {
        let node = dag.add_node(DataVertex::OpResult(inst.clone()));

        for i in 0..n {
            let arg = stack.pop().right_or_else(|var| dag.add_node(var));
            dag.add_edge(arg, node, i).unwrap();
        }

        stack.push(node);
    }

    for inst in code {
        match inst {
            LinInst::Const(_) => n_arity_inst(&mut dag, &mut stack, 0, inst),
            LinInst::Elem => n_arity_inst(&mut dag, &mut stack, 2, inst),
            LinInst::Store(_) => todo!(),
            LinInst::Load(_) => todo!(),
            LinInst::Call(_, n) => n_arity_inst(&mut dag, &mut stack, n, inst),
            LinInst::BinOp(_) => n_arity_inst(&mut dag, &mut stack, 2, inst),
            LinInst::Tag(_, _) => n_arity_inst(&mut dag, &mut stack, 1, inst),
            LinInst::SExp(_, n) => n_arity_inst(&mut dag, &mut stack, n, inst),
            LinInst::Dup => {
                let index = stack.peek().right_or_else(|var| dag.add_node(var));
                stack.push(index);
            }
            LinInst::Drop => {
                stack.pop();
            }
        }
    }

    let inputs = dag
        .node_references()
        .filter_map(|(i, typ)| match typ {
            DataVertex::StackVar(_) => Some(i),
            DataVertex::OpResult(_) => None,
        })
        .collect();

    DataGraph {
        dag,
        inputs,
        outputs: stack.stack,
    }
}

fn compile_function(code: Vec<Inst>) -> FlowGraph {
    let mut graph = Graph::new();
    let mut by_label: BTreeMap<Ident, NodeIndex> = BTreeMap::new();
    let mut by_line: BTreeMap<usize, NodeIndex> = BTreeMap::new();
    let mut block_endings: Vec<(NodeIndex, FlowInst, usize)> = Vec::new();

    let flow_insts: Vec<_> = code
        .iter()
        .enumerate()
        .filter_map(|(i, inst)| match inst {
            Inst::LinInst(_) => None,
            Inst::FlowInst(_) => Some(i),
        })
        .collect();

    let linear_blocks = flow_insts.iter().cloned().tuple_windows();
    // .filter(|(a, b)| b - a > 1);

    for (prev_flow, end) in linear_blocks {
        let start = prev_flow + 1;
        let block = code[start..end]
            .iter()
            .filter_map(|inst| match inst {
                Inst::LinInst(lin_inst) => Some(lin_inst.clone()),
                Inst::FlowInst(_) => None,
            })
            .collect();

        let block = compile_lin_block(block);
        let node = graph.add_node(block);

        if let Inst::FlowInst(FlowInst::Label(id)) = &code[prev_flow] {
            by_label.insert(id.clone(), node);
        }

        by_line.insert(start, node);

        let final_instruction = match &code[end] {
            Inst::FlowInst(flow_inst) => flow_inst.clone(),
            Inst::LinInst(_) => unreachable!(),
        };

        block_endings.push((node, final_instruction, end));
    }

    for (node, ending, end_line) in block_endings {
        match ending {
            FlowInst::Jmp(jump_mode, label) => {
                let jump_cont = by_label[&label];
                let jump_condition = match jump_mode {
                    JumpMode::Unconditional => JumpCondition::Unconditional,
                    JumpMode::Zero => JumpCondition::Zero,
                    JumpMode::NonZero => JumpCondition::NonZero,
                };

                graph.add_edge(node, jump_cont, jump_condition);

                match jump_mode {
                    JumpMode::Zero => {
                        let (_, fallthrough_cont) = by_line.range(end_line..).next().unwrap();
                        graph.add_edge(node, *fallthrough_cont, JumpCondition::NonZero);
                    }
                    JumpMode::NonZero => {
                        let (_, fallthrough_cont) = by_line.range(end_line..).next().unwrap();
                        graph.add_edge(node, *fallthrough_cont, JumpCondition::Zero);
                    }
                    _ => {}
                }
            }
            FlowInst::Label(label) => {
                let cont = by_label[&label];
                graph.add_edge(node, cont, JumpCondition::Unconditional);
            }
            FlowInst::Ret => todo!(),
            _ => {}
        }
    }

    graph
}

struct Escaper<W>(W);

impl<W> fmt::Write for Escaper<W>
where
    W: fmt::Write,
{
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for c in s.chars() {
            self.write_char(c)?;
        }

        Ok(())
    }

    fn write_char(&mut self, c: char) -> fmt::Result {
        match c {
            '"' | '\\' => self.0.write_char('\\')?,

            // \l is for left justified linebreak
            '\n' => return self.0.write_str("\\l"),

            _ => {}
        }

        self.0.write_char(c)
    }
}

fn subgraph<W: Write>(w: &mut W, label: &str, graph: &DataGraph) -> fmt::Result {
    writeln!(w, "subgraph cluster_{label} {{")?;
    writeln!(w, "label = \"{label}\";")?;
    writeln!(w, "style = filled;")?;
    writeln!(w, "color = lightgrey;")?;
    writeln!(w, "sub{label}_input;")?;
    writeln!(w, "sub{label}_output;")?;

    for node in graph.inputs.iter() {
        writeln!(w, "sub{label}_input -> sub{label}{};", node.index())?;
    }

    for node in graph.outputs.iter() {
        writeln!(w, "sub{label}{} -> sub{label}_output;", node.index())?;
    }

    for node in graph.dag.graph().node_indices() {
        write!(w, "sub{label}{} [label = \"", node.index(),)?;
        write!(Escaper(&mut *w), "{:?}", &graph.dag[node])?;
        writeln!(w, "\"];")?;
    }

    for edge in graph.dag.graph().edge_references() {
        writeln!(
            w,
            "sub{label}{} -> sub{label}{} [label = \"{}\"];",
            edge.source().index(),
            edge.target().index(),
            edge.weight()
        )?;
    }

    writeln!(w, "}}")?;
    Ok(())
}

fn function_graph<W: Write>(w: &mut W, graph: &FlowGraph) -> fmt::Result {
    writeln!(w, "digraph {{")?;

    for block in graph.node_indices() {
        let label = block.index().to_string();
        subgraph(w, &label, &graph[block])?;
    }

    for edge in graph.edge_references() {
        writeln!(
            w,
            "sub{}_output -> sub{}_input [label = \"{:?}\"]",
            edge.source().index(),
            edge.target().index(),
            edge.weight()
        )?;
    }

    writeln!(w, "}}")?;
    Ok(())
}

fn main() {
    // let code = Vec::from([
    //     LinInst::Const(10),
    //     LinInst::BinOp(Op::Plus),
    //     LinInst::BinOp(Op::Minus),
    //     LinInst::Const(20),
    //     LinInst::Call(String::from("foo"), 3),
    //     LinInst::Dup,
    //     LinInst::SExp(String::from("tag"), 2),
    // ]);

    let code = Vec::from([
        Inst::FlowInst(FlowInst::Begin),
        Inst::LinInst(LinInst::Const(10)),
        Inst::LinInst(LinInst::Const(20)),
        Inst::LinInst(LinInst::BinOp(Op::Plus)),
        Inst::FlowInst(FlowInst::Jmp(JumpMode::NonZero, String::from("nonzero"))),
        Inst::LinInst(LinInst::SExp(String::from("Foo"), 3)),
        Inst::FlowInst(FlowInst::Jmp(JumpMode::Unconditional, String::from("exit"))),
        Inst::FlowInst(FlowInst::Label(String::from("nonzero"))),
        Inst::LinInst(LinInst::SExp(String::from("Bar"), 3)),
        Inst::FlowInst(FlowInst::Label(String::from("exit"))),
        Inst::FlowInst(FlowInst::End),
    ]);

    let output = compile_function(code);

    let mut buffer = String::new();
    function_graph(&mut buffer, &output).unwrap();
    println!("{buffer}");
}
