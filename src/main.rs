use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Write};
use std::fs::File;
use std::io::read_to_string;

use daggy::{Dag, NodeIndex};
use either::Either::{self, Left, Right};
use itertools::Itertools;
use petgraph::dot::Config::GraphContentOnly;
use petgraph::visit::{EdgeRef, IntoNodeReferences, NodeRef};
use petgraph::{dot::Dot, Graph};
use Inst::*;

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
    start_label: String,
    dag: Dag<DataVertex, ArgStackOffset>,
    inputs: Vec<NodeIndex>,
    outputs: Vec<NodeIndex>,
}

impl DataGraph {
    fn info_label(&self) -> String {
        format!(
            "{}\\linputs: {}\\loutputs: {}",
            self.start_label,
            self.inputs.len(),
            self.outputs.len()
        )
    }
}

#[derive(Debug, Clone, Copy)]
enum JumpCondition {
    Unconditional,
    Zero,
    NonZero,
}

#[derive(Debug, Clone)]
enum DataVertex {
    Symbolic(Ident),
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

fn inst_node(dag: &mut Dag<DataVertex, ArgStackOffset>, inst: LinInst) -> NodeIndex {
    dag.add_node(DataVertex::OpResult(inst))
}

fn compile_lin_block(start_label: String, code: Vec<LinInst>) -> DataGraph {
    let mut dag = Dag::new();
    let mut stack = VirtualStack::new();
    let mut symbolic: BTreeMap<Ident, NodeIndex> = BTreeMap::new();

    fn n_arity_inst(
        dag: &mut Dag<DataVertex, usize>,
        stack: &mut VirtualStack,
        n: usize,
        inst: LinInst,
    ) {
        let node = inst_node(dag, inst);

        for i in 0..n {
            let arg = stack.pop().right_or_else(|var| dag.add_node(var));
            dag.add_edge(arg, node, i).unwrap();
        }

        stack.push(node);
    }

    for inst in code {
        // Assume, that all of the instructions consume stack values, aside from ST
        match inst {
            LinInst::Const(_) => n_arity_inst(&mut dag, &mut stack, 0, inst),
            LinInst::Elem => n_arity_inst(&mut dag, &mut stack, 2, inst),
            LinInst::Store(ref name) => {
                let node = inst_node(&mut dag, inst.clone());
                symbolic.insert(name.clone(), node);
            }
            LinInst::Load(ref name) => {
                if !symbolic.contains_key(name) {
                    let node = dag.add_node(DataVertex::Symbolic(name.clone()));
                    symbolic.insert(name.clone(), node);
                }

                stack.push(symbolic[name]);
            }
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
            DataVertex::Symbolic(_) => Some(i),
            DataVertex::OpResult(_) => None,
        })
        .collect();

    DataGraph {
        start_label,
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

    for (prev_flow, end) in linear_blocks {
        eprintln!("handle block from {} to {end}", prev_flow + 1);

        let start = prev_flow + 1;
        let block = code[start..end]
            .iter()
            .filter_map(|inst| match inst {
                Inst::LinInst(lin_inst) => Some(lin_inst.clone()),
                Inst::FlowInst(_) => None,
            })
            .collect();

        let start_label = match &code[prev_flow] {
            Inst::LinInst(_) => unreachable!(),
            Inst::FlowInst(FlowInst::Label(label)) => label.clone(),
            _ => format!("line {}", prev_flow + 1),
        };

        let block = compile_lin_block(start_label, block);
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
    writeln!(w, "label = \"{}\";", graph.info_label())?;
    writeln!(w, "style = filled;")?;
    writeln!(w, "color = lightgrey;")?;
    writeln!(w, "sub{label}_input [shape = point style = invis];")?;
    writeln!(w, "sub{label}_output [shape = point style = invis];")?;

    for node in graph.inputs.iter() {
        writeln!(w, "sub{label}_input -> sub{label}{};", node.index())?;
    }

    for node in graph.outputs.iter() {
        writeln!(w, "sub{label}{} -> sub{label}_output;", node.index())?;
    }

    for node in graph.dag.graph().node_indices() {
        write!(w, "sub{label}{} [label = \"", node.index(),)?;
        write!(Escaper(&mut *w), "{:?}", &graph.dag[node])?;
        writeln!(
            w,
            "\" {}];",
            if graph.outputs.contains(&node) {
                "color = red"
            } else {
                ""
            }
        )?;
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
    writeln!(w, "digraph G {{")?;
    writeln!(w, "compound = true;")?;

    for block in graph.node_indices() {
        let label = block.index().to_string();
        subgraph(w, &label, &graph[block])?;
    }

    for edge in graph.edge_references() {
        writeln!(
            w,
            "sub{}_output -> sub{}_input [label = \"{:?}\" ltail = cluster_{} lhead=cluster_{}]",
            edge.source().index(),
            edge.target().index(),
            edge.weight(),
            edge.source().index(),
            edge.target().index(),
        )?;
    }

    writeln!(w, "}}")?;
    Ok(())
}

fn const_i(v: i32) -> Inst {
    LinInst(LinInst::Const(v))
}

fn bin_op(op: Op) -> Inst {
    LinInst(LinInst::BinOp(op))
}

fn jmp(mode: JumpMode, label: &str) -> Inst {
    FlowInst(FlowInst::Jmp(mode, label.to_string()))
}

fn elem() -> Inst {
    LinInst(LinInst::Elem)
}

fn label(l: &str) -> Inst {
    FlowInst(FlowInst::Label(l.to_string()))
}

fn dup() -> Inst {
    LinInst(LinInst::Dup)
}

fn drop_i() -> Inst {
    LinInst(LinInst::Drop)
}

fn load(l: &str) -> Inst {
    LinInst(LinInst::Load(l.to_string()))
}

fn store(l: &str) -> Inst {
    LinInst(LinInst::Store(l.to_string()))
}

fn tag(t: &str, n: usize) -> Inst {
    LinInst(LinInst::Tag(t.to_string(), n))
}

fn sexp(t: &str, n: usize) -> Inst {
    LinInst(LinInst::SExp(t.to_string(), n))
}

fn linear_example() -> Vec<LinInst> {
    Vec::from([
        LinInst::Const(10),
        LinInst::BinOp(Op::Plus),
        LinInst::BinOp(Op::Minus),
        LinInst::Const(20),
        LinInst::Call(String::from("foo"), 3),
        LinInst::Dup,
        LinInst::SExp(String::from("tag"), 2),
    ])
}

fn simple_example() -> Vec<Inst> {
    Vec::from([
        FlowInst(FlowInst::Begin),            // 0
        const_i(10),                          // 1
        const_i(20),                          // 2
        bin_op(Op::Plus),                     // 3
        jmp(JumpMode::NonZero, "nonzero"),    // 4
        sexp("Foo", 3),                       // 5
        jmp(JumpMode::Unconditional, "exit"), // 6
        label("nonzero"),                     // 7
        sexp("Bar", 3),                       // 8
        label("exit"),                        // 9
        const_i(1),                           // 10
        elem(),                               // 11
        FlowInst(FlowInst::End),              // 12
    ])
}

fn parse_stack_code(code: &str) -> Vec<Inst> {
    code.lines()
        .map(|line| {
            let mut tokens = line
                .split(|ch| match ch {
                    ',' | '(' | ')' => true,
                    _ if ch.is_whitespace() => true,
                    _ => false,
                })
                .filter(|token| !token.is_empty());

            match tokens.next() {
                Some("BEGIN") => FlowInst(FlowInst::Begin),
                Some("END") => FlowInst(FlowInst::End),
                Some("LD") => {
                    let place = tokens.next().unwrap().to_string();
                    LinInst(LinInst::Load(place))
                }
                Some("ST") => {
                    let place = tokens.next().unwrap().to_string();
                    LinInst(LinInst::Load(place))
                }
                Some("DUP") => LinInst(LinInst::Dup),
                Some("DROP") => LinInst(LinInst::Drop),
                Some("LABEL") => {
                    let label = tokens.next().unwrap().to_string();
                    FlowInst(FlowInst::Label(label))
                }
                Some("CONST") => {
                    let value = tokens.next().unwrap().parse().unwrap();
                    LinInst(LinInst::Const(value))
                }
                Some("JMP") => {
                    let label = tokens.next().unwrap().to_string();
                    FlowInst(FlowInst::Jmp(JumpMode::Unconditional, label))
                }
                Some("CJMP") => {
                    let mode = match tokens.next() {
                        Some("z") => JumpMode::Zero,
                        Some("nz") => JumpMode::NonZero,
                        _ => panic!("unknown jump command"),
                    };

                    let label = tokens.next().unwrap().to_string();
                    FlowInst(FlowInst::Jmp(mode, label))
                }
                Some("ELEM") => LinInst(LinInst::Elem),
                Some("PATT") => {
                    let _ = tokens.next();
                    let tag = tokens
                        .next()
                        .and_then(|s| s.strip_prefix('\"'))
                        .and_then(|s| s.strip_suffix('\"'))
                        .unwrap();

                    let num = tokens.next().unwrap().parse().unwrap();
                    LinInst(LinInst::Tag(tag.to_string(), num))
                }
                Some("SEXP") => todo!(),
                Some("CALL") => todo!(),
                _ => panic!("unknown command"),
            }
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let content = read_to_string(File::open("head.sm")?)?;
    let code = parse_stack_code(&content);
    let output = compile_function(code);
    let mut buffer = String::new();
    function_graph(&mut buffer, &output).unwrap();
    println!("{buffer}");

    Ok(())
}
