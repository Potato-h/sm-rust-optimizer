pub mod analyze;
pub mod compile;
pub mod optimize;

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::{self, Display, Write};
use std::{iter, mem};

use crate::common::{self, Inst::Flow, Inst::Linear, *};
use either::Either::{self, Left, Right};
use itertools::Itertools;
use petgraph::csr::IndexType;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences};
use petgraph::{algo, prelude::*, EdgeType};

// FIXME: in general case outputs and jump_decided_by may overlap
// it can break merging of blocks
#[derive(Debug, Clone)]
pub struct DataGraph {
    pub start_label: String,
    pub dag: StableGraph<DataVertex, ArgStackOffset>,
    pub inputs: Vec<NodeIndex>,
    pub outputs: VirtualStack,
    pub symbolics: BTreeMap<Sym, NodeIndex>,
    pub jump_decided_by: Option<NodeIndex>,
}

impl DataGraph {
    pub fn outputs_nodes(&self) -> &[NodeIndex] {
        &self.outputs.stack
    }

    pub fn info_label(&self) -> String {
        format!(
            "{}\ninputs: {}\noutputs: {}",
            self.start_label,
            self.inputs.len(),
            self.outputs_nodes().len()
        )
    }

    pub fn args_count(&self) -> u16 {
        self.symbolics
            .keys()
            .filter_map(|x| match x {
                Sym::Arg(x) => Some(x + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }

    pub fn locs_count(&self) -> u16 {
        self.symbolics
            .keys()
            .filter_map(|x| match x {
                Sym::Loc(x) => Some(x + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }

    pub fn clos_count(&self) -> u16 {
        self.symbolics
            .keys()
            .filter_map(|x| match x {
                Sym::Acc(x) => Some(x + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }

    pub fn check_invariants(&self) {
        for input in self.inputs.iter() {
            assert!(
                self.dag.contains_node(*input),
                "input node in data graph must be alive"
            );
        }

        for output in self.outputs.stack.iter() {
            assert!(
                self.dag.contains_node(*output),
                "output node in data graph must be alive"
            );
        }

        for jump in self.jump_decided_by.iter() {
            assert!(
                self.dag.contains_node(*jump),
                "jump node in data graph must be alive"
            );
        }

        assert!(
            !algo::is_cyclic_directed(&self.dag),
            "found cycle in data flow graph"
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JumpCondition {
    Unconditional,
    Zero,
    NonZero,
}

impl JumpCondition {
    pub fn will_jump(&self, value: i32) -> bool {
        match self {
            JumpCondition::Unconditional => true,
            JumpCondition::Zero if value == 0 => true,
            JumpCondition::NonZero if value != 0 => true,
            _ => false,
        }
    }

    pub fn is_conditional(&self) -> bool {
        matches!(self, JumpCondition::NonZero | JumpCondition::Zero)
    }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataVertex {
    Symbolic(Sym),
    StackVar(ArgStackOffset),
    OpResult(LinInst),
}

impl Display for DataVertex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataVertex::Symbolic(sym) => write!(f, "{sym}"),
            DataVertex::StackVar(offset) => write!(f, "StackVar({offset})"),
            DataVertex::OpResult(inst) => write!(f, "{inst}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtualStack {
    pub tail_from: usize,
    pub stack: Vec<NodeIndex>,
}

impl VirtualStack {
    pub fn new() -> Self {
        VirtualStack {
            tail_from: 0,
            stack: Vec::new(),
        }
    }

    pub fn push(&mut self, v: NodeIndex) {
        self.stack.push(v);
    }

    pub fn pop(&mut self) -> Either<DataVertex, NodeIndex> {
        match self.stack.pop() {
            Some(v) => Right(v),
            None => {
                let tail = self.tail_from;
                self.tail_from += 1;
                Left(DataVertex::StackVar(tail))
            }
        }
    }

    pub fn peek(&self) -> Either<DataVertex, NodeIndex> {
        match self.stack.last() {
            Some(&v) => Right(v),
            None => {
                let tail = self.tail_from;
                Left(DataVertex::StackVar(tail))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct CallVertex {
    pub label: String,
    pub name: String,
    pub args: usize,
}

#[derive(Debug, Clone)]
pub struct STIVertex {
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct STAVertex {
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct CallCVertex {
    pub label: String,
    pub args: usize,
}

#[derive(Debug, Clone)]
pub enum FlowVertex {
    LinearBlock(DataGraph),
    Call(CallVertex),
    STI(STIVertex),
    STA(STAVertex),
    CallC(CallCVertex),
}

impl FlowVertex {
    pub fn is_block(&self) -> bool {
        matches!(self, Self::LinearBlock(_))
    }

    pub fn block(&self) -> Option<&DataGraph> {
        match self {
            FlowVertex::LinearBlock(graph) => Some(graph),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlowGraph {
    pub graph: StableGraph<FlowVertex, JumpCondition>,
    pub input: NodeIndex,
    pub outputs: Vec<NodeIndex>,
}

/// Insert graph `extend` into graph `source` and return mapping of old `extend` indexes
/// into indexes into updated `source`.
fn add_graph<N, E, Ty, Ix>(
    source: &mut StableGraph<N, E, Ty, Ix>,
    extend: &StableGraph<N, E, Ty, Ix>,
) -> BTreeMap<NodeIndex<Ix>, NodeIndex<Ix>>
where
    Ix: IndexType,
    Ty: EdgeType,
    N: Clone,
    E: Clone,
{
    let mut extend_to_source = BTreeMap::new();

    for (id, w) in extend.node_references() {
        let new_id = source.add_node(w.clone());
        extend_to_source.insert(id, new_id);
    }

    for edge in extend.edge_references() {
        source.add_edge(
            extend_to_source[&edge.source()],
            extend_to_source[&edge.target()],
            edge.weight().clone(),
        );
    }

    extend_to_source
}

#[derive(Debug, Clone)]
pub struct JmpEdges {
    pub jmp: NodeIndex,
    pub cjmp: Option<(JumpMode, NodeIndex)>,
}

impl FlowGraph {
    pub fn args_count(&self) -> u16 {
        self.graph
            .node_weights()
            .filter_map(|v| v.block().map(DataGraph::args_count))
            .max()
            .unwrap_or(0)
    }

    pub fn loc_count(&self) -> u16 {
        self.graph
            .node_weights()
            .filter_map(|v| v.block().map(DataGraph::locs_count))
            .max()
            .unwrap_or(0)
    }

    pub fn clos_count(&self) -> u16 {
        self.graph
            .node_weights()
            .filter_map(|v| v.block().map(DataGraph::clos_count))
            .max()
            .unwrap_or(0)
    }

    pub fn get_label(&self, node: NodeIndex) -> String {
        match &self.graph[node] {
            FlowVertex::LinearBlock(block) => block.start_label.clone(),
            FlowVertex::Call(call) => call.label.clone(),
            FlowVertex::STI(sti) => sti.label.clone(),
            FlowVertex::STA(sta) => sta.label.clone(),
            FlowVertex::CallC(callc) => callc.label.clone(),
        }
    }

    /// Since flow graph forget actual jumps information from stack machine code,
    /// it's useful to provide helper that will reify it back
    pub fn jmp_outgoings(&self, node: NodeIndex) -> Option<JmpEdges> {
        let mut jmp = None;
        let mut cjmp = None;

        // FIXME: dirty hack -- now that there is only one unconditional jump or
        // 2 jumps with different conditions. This loop occurs because of poor
        // choice of graph representation.
        for (i, edge) in self
            .graph
            .edges_directed(node, Direction::Outgoing)
            .enumerate()
        {
            match edge.weight() {
                _ if i > 0 => {
                    jmp = Some(edge.target());
                }
                JumpCondition::Unconditional => {
                    jmp = Some(edge.target());
                }
                JumpCondition::Zero => {
                    cjmp = Some((JumpMode::Zero, edge.target()));
                }
                JumpCondition::NonZero => {
                    cjmp = Some((JumpMode::NonZero, edge.target()));
                }
            };
        }

        Some(JmpEdges { jmp: jmp?, cjmp })
    }

    pub fn has_calls(&self) -> bool {
        self.graph
            .node_weights()
            .any(|n| matches!(n, FlowVertex::Call(_) | FlowVertex::CallC(_)))
    }

    pub fn all_symbolics(&self) -> BTreeSet<Sym> {
        self.graph
            .node_weights()
            .filter_map(|v| match v {
                // TODO: probably only need to collect input symbolics?
                FlowVertex::LinearBlock(block) => Some(block.symbolics.keys()),
                _ => None,
            })
            .flatten()
            .cloned()
            .collect()
    }

    // TODO: precise analysis of CALLs
    // TODO: precise analysis of STA and STI
    pub fn find_read_symbolics(&self) -> BTreeMap<NodeIndex, BTreeSet<Sym>> {
        let mut read_syms = BTreeMap::new();
        let all_symbolics = self.all_symbolics();

        // symbolic variables that needed to be preserved across function calls
        // does not need to preserve Acc and Arg, because only way to modify them
        // is to call STA, but it's already a separate vertex can not be deleted
        let globals: BTreeSet<_> = all_symbolics
            .iter()
            .filter(|v| matches!(v, Sym::Glb(_)))
            .cloned()
            .collect();

        for &output in self.outputs.iter() {
            read_syms.insert(output, globals.clone());
        }

        // FIXME: this is wrong for many reasons
        // 1) cycles breaks this approach (SCC or known liveness analysis can solve this)
        // 2) it doesn't respect CALL, CALLC, STI and STA
        // 3) may need an interval (or more complex structure) for each symbolic variable,
        // not variables for each flow node
        fn dfs(
            flow: &FlowGraph,
            node: NodeIndex,
            visited: &mut BTreeSet<NodeIndex>,
            read_syms: &mut BTreeMap<NodeIndex, BTreeSet<Sym>>,
            all_symbolics: &BTreeSet<Sym>,
        ) {
            visited.insert(node);

            let syms = match &flow.graph[node] {
                FlowVertex::LinearBlock(block) => block
                    .inputs
                    .iter()
                    .filter_map(|&v| match &block.dag[v] {
                        DataVertex::Symbolic(sym) => Some(sym.clone()),
                        _ => None,
                    })
                    .collect(),
                _ => all_symbolics.clone(),
            };

            read_syms.entry(node).or_default().extend(syms);

            // TODO: use new disjoin API on BTreeMap
            for target in flow.graph.neighbors_directed(node, Direction::Outgoing) {
                if !visited.contains(&target) {
                    dfs(flow, target, visited, read_syms, all_symbolics);
                }

                let ext = read_syms[&target].iter().cloned().collect_vec();

                if let Some(syms) = read_syms.get_mut(&node) {
                    syms.extend(ext);
                }
            }
        }

        let mut visited = BTreeSet::new();
        for node in self.graph.node_indices() {
            visited.clear();
            dfs(self, node, &mut visited, &mut read_syms, &all_symbolics);
        }

        read_syms
    }

    pub fn find_used_decls(&self) -> BTreeSet<Ident> {
        let mut decls = BTreeSet::new();

        for node in self.graph.node_weights() {
            match node {
                FlowVertex::LinearBlock(block) => {
                    for node in block.dag.node_weights() {
                        if let DataVertex::OpResult(LinInst::Closure(name, _)) = node {
                            decls.insert(name.clone());
                        }
                    }
                }
                FlowVertex::Call(call) => {
                    decls.insert(call.name.clone());
                }
                _ => {}
            }
        }

        decls
    }
}

pub fn write_code<W: Write>(w: &mut W, code: &[Inst]) -> fmt::Result {
    for inst in code.iter() {
        writeln!(w, "{inst}")?;
    }

    writeln!(w, "!!")?;

    Ok(())
}

fn gen_inline_call_prologue(label: String, args: u16, first_free_loc: u16) -> DataGraph {
    let inst = (0..args)
        .rev()
        .flat_map(|i| [LinInst::Store(Sym::Loc(first_free_loc + i)), LinInst::Drop])
        .collect_vec();

    DataGraph::analyze(label, inst, false)
}

fn gen_tail_call_prologue(label: Label, args: u16) -> DataGraph {
    let code = (0..args)
        .rev()
        .flat_map(|i| [LinInst::Store(Sym::Arg(i)), LinInst::Drop])
        .collect_vec();

    DataGraph::analyze(label, code, false)
}

pub struct UnitOptimFlags {
    pub flow_optim: FlowOptimFlags,
    pub force_inline: Vec<String>,
    pub inline_strategy: bool,
    pub remove_unused_decls: bool,
    pub passes: u32,
}

#[derive(Debug)]
pub struct Unit {
    pub functions: BTreeMap<Ident, FlowGraph>,
    pub declarations: Vec<Decl>,
}
