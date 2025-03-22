use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{self, Write};
use std::fs::File;
use std::io::read_to_string;
use std::mem;

use either::Either::{self, Left, Right};
use itertools::Itertools;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences};
use petgraph::{algo, prelude::*};
use Inst::*;

type Ident = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum JumpMode {
    Unconditional,
    Zero,
    NonZero,
}

impl JumpMode {
    fn rev(self) -> Self {
        match self {
            JumpMode::Unconditional => JumpMode::Unconditional,
            JumpMode::Zero => JumpMode::NonZero,
            JumpMode::NonZero => JumpMode::Zero,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Plus,
    Minus,
    Eq,
}

#[derive(Debug, Clone)]
enum LinInst {
    Const(i32),
    Elem,
    Store(Ident),
    Load(Ident),
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
    // call command should be in flow graph, because it can cause
    // arbitrary change of global variables. And after unfolding it can
    // have control flow instruction which weird to extract from lin block
    Call(Ident, usize),
    Ret,
    Begin,
    End,
}

impl FlowInst {
    fn conditional_jmp(&self) -> bool {
        matches!(
            self,
            FlowInst::Jmp(JumpMode::NonZero, _) | FlowInst::Jmp(JumpMode::Zero, _)
        )
    }
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
    dag: StableGraph<DataVertex, ArgStackOffset>,
    inputs: Vec<NodeIndex>,
    outputs: Vec<NodeIndex>,
    jump_decided_by: Option<NodeIndex>,
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

impl From<JumpMode> for JumpCondition {
    fn from(value: JumpMode) -> Self {
        match value {
            JumpMode::Unconditional => JumpCondition::Unconditional,
            JumpMode::Zero => JumpCondition::Zero,
            JumpMode::NonZero => JumpCondition::NonZero,
        }
    }
}

#[derive(Debug, Clone)]
enum DataVertex {
    Symbolic(Ident),
    StackVar(usize),
    OpResult(LinInst),
}

#[derive(Debug)]
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
            Some(&v) => Right(v),
            None => {
                let tail = self.tail_from;
                Left(DataVertex::StackVar(tail))
            }
        }
    }
}

enum FlowVertex {
    LinearBlock(DataGraph),
    Call(String, usize),
}

struct FlowGraph {
    graph: StableGraph<FlowVertex, JumpCondition>,
    input: NodeIndex,
    outputs: Vec<NodeIndex>,
}

fn inst_vertex(inst: &LinInst) -> DataVertex {
    DataVertex::OpResult(inst.clone())
}

fn analyze_lin_block(start_label: String, code: Vec<LinInst>, has_cjmp: bool) -> DataGraph {
    let mut dag = StableGraph::new();
    let mut stack = VirtualStack::new();
    let mut symbolic: BTreeMap<Ident, NodeIndex> = BTreeMap::new();

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
            LinInst::Elem => n_arity_inst(&mut dag, &mut stack, 2, inst),
            LinInst::Store(ref name) => {
                let node = dag.add_node(inst_vertex(&inst));
                let from = stack.peek().right_or_else(|var| dag.add_node(var));
                dag.add_edge(from, node, 0);
                symbolic.insert(name.clone(), node);
            }
            LinInst::Load(ref name) => {
                if !symbolic.contains_key(name) {
                    let node = dag.add_node(DataVertex::Symbolic(name.clone()));
                    symbolic.insert(name.clone(), node);
                }

                stack.push(symbolic[name]);
            }
            LinInst::BinOp(_) => n_arity_inst(&mut dag, &mut stack, 2, inst),
            LinInst::Tag(_, _) => n_arity_inst(&mut dag, &mut stack, 1, inst),
            LinInst::SExp(_, n) => n_arity_inst(&mut dag, &mut stack, n, inst),
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
                stack.pop();
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

    DataGraph {
        start_label,
        dag,
        inputs,
        outputs: stack.stack,
        jump_decided_by,
    }
}

impl DataGraph {
    // TODO: is removal of input node breaks relation of block? probably not,
    // because it's only matters for stack values, but we annotate stack depth
    // for each node.
    // TODO: is some cases symbolic variable may be useless for specific block,
    // but actually bypassed for next blocks
    fn eliminate_dead_code(&mut self) {
        let outputs: BTreeSet<_> = self
            .jump_decided_by
            .iter()
            .chain(self.outputs.iter())
            .cloned()
            .collect();

        while let Some(sink) = self
            .dag
            .node_indices()
            .filter(|id| !outputs.contains(id))
            .find(|id| self.dag.edges_directed(*id, Direction::Outgoing).count() == 0)
        {
            self.dag.remove_node(sink);
            self.inputs.retain(|&x| x != sink);
        }
    }

    fn constant_propagation(&mut self) {
        todo!()
    }

    fn optimize(&mut self) {
        self.eliminate_dead_code();
    }
}

type Label = String;

impl FlowGraph {
    fn eliminate_dead_code(&mut self) {
        while let Some(source) = self
            .graph
            .node_indices()
            .filter(|&id| id != self.input)
            .find(|&id| self.graph.edges_directed(id, Direction::Incoming).count() == 0)
        {
            self.graph.remove_node(source);
        }
    }

    fn optimize(&mut self) {
        for block in self.graph.node_weights_mut() {
            match block {
                FlowVertex::LinearBlock(graph) => graph.optimize(),
                FlowVertex::Call(_, _) => {}
            }
        }

        self.eliminate_dead_code();
    }

    // TODO: fill all outputs (is RET operation a early return or something else?)
    fn analyze_function(code: Vec<Inst>) -> Self {
        let mut edges: Vec<(NodeIndex, Label, JumpCondition)> = Vec::new();
        let mut edge_from_prev: Option<(NodeIndex, JumpCondition)> = None;
        let mut block: Vec<LinInst> = Vec::new();
        let mut graph = StableGraph::new();
        let mut labeled: Option<Label> = None;
        let mut block_indexes = Vec::new(); // only purpose is to find input and outputs blocks
        let mut from_label_to_block: BTreeMap<Label, NodeIndex> = BTreeMap::new();

        // skip(1) for Begin operation
        for (i, inst) in code.into_iter().enumerate().skip(1) {
            match inst {
                Inst::LinInst(inst) => {
                    block.push(inst);
                }
                Inst::FlowInst(inst) => {
                    let start_label = labeled.take().unwrap_or(format!("Line_{i}"));
                    let has_cjmp = inst.conditional_jmp();
                    let this_block =
                        analyze_lin_block(start_label.clone(), mem::take(&mut block), has_cjmp);

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
                        FlowInst::Label(label) => {
                            edge_from_prev = Some((node, JumpCondition::Unconditional));
                            labeled = Some(label);
                        }
                        FlowInst::Call(name, args) => {
                            let call_node = graph.add_node(FlowVertex::Call(name, args));
                            graph.add_edge(node, call_node, JumpCondition::Unconditional);
                            edge_from_prev = Some((call_node, JumpCondition::Unconditional));
                        }
                        FlowInst::Ret => todo!(),
                        FlowInst::Begin => continue,
                        FlowInst::End => break,
                    }
                }
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

    for node in graph.dag.node_indices() {
        write!(w, "sub{label}{} [label = \"", node.index(),)?;
        write!(Escaper(&mut *w), "{:?}", &graph.dag[node])?;
        writeln!(
            w,
            "\" {}];",
            match () {
                _ if graph.inputs.contains(&node) && graph.outputs.contains(&node) =>
                    "color = yellow",
                _ if graph.inputs.contains(&node) => "color = green",
                _ if graph.outputs.contains(&node) => "color = red",
                _ if graph.jump_decided_by.iter().contains(&node) => "color = purple",
                _ => "",
            }
        )?;
    }

    for edge in graph.dag.edge_references() {
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

fn call_subgraph<W: Write>(w: &mut W, label: &str, name: &str, args: usize) -> fmt::Result {
    writeln!(w, "subgraph cluster_{label} {{")?;
    writeln!(w, "label = \"call ({}, {})\";", name, args)?;
    writeln!(w, "style = filled;")?;
    writeln!(w, "color = lightgrey;")?;
    writeln!(w, "sub{label}_input [shape = point style = invis];")?;
    writeln!(w, "sub{label}_output [shape = point style = invis];")?;
    writeln!(w, "}}")?;
    Ok(())
}

fn function_graph<W: Write>(w: &mut W, flow: &FlowGraph) -> fmt::Result {
    writeln!(w, "digraph G {{")?;
    writeln!(w, "compound = true;")?;

    for block in flow.graph.node_indices() {
        let label = block.index().to_string();
        match &flow.graph[block] {
            FlowVertex::LinearBlock(graph) => subgraph(w, &label, graph)?,
            FlowVertex::Call(name, args) => call_subgraph(w, &label, name, *args)?,
        }
    }

    for edge in flow.graph.edge_references() {
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
                    LinInst(LinInst::Store(place))
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
                Some("BINOP") => match tokens.next() {
                    Some("==") => LinInst(LinInst::BinOp(Op::Eq)),
                    x => panic!("unknown binary op: {x:?}"),
                },
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
                Some("SEXP") => {
                    let tag = tokens
                        .next()
                        .and_then(|s| s.strip_prefix('\"'))
                        .and_then(|s| s.strip_suffix('\"'))
                        .unwrap();

                    let num = tokens.next().unwrap().parse().unwrap();
                    LinInst(LinInst::SExp(tag.to_string(), num))
                }
                Some("CALL") => {
                    let name = tokens.next().unwrap().to_string();
                    let num = tokens.next().unwrap().parse().unwrap();
                    FlowInst(FlowInst::Call(name, num))
                }
                _ => panic!("unknown command"),
            }
        })
        .collect()
}

// TODO: make cli
fn main() -> Result<(), Box<dyn Error>> {
    let content = read_to_string(File::open("concat.sm")?)?;
    let code = parse_stack_code(&content);
    let mut output = FlowGraph::analyze_function(code);
    output.optimize();
    let mut buffer = String::new();
    function_graph(&mut buffer, &output).unwrap();
    println!("{buffer}");

    Ok(())
}
