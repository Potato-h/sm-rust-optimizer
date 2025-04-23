use std::error::Error;
use std::fs::{self, File};
use std::io::read_to_string;
use std::path::PathBuf;

use clap::Parser;
use sm_optimizer::common::*;
use sm_optimizer::core::{write_code, Unit, UnitOptimFlags};
use sm_optimizer::dot::function_graph;

fn parse_stack_code(code: &str) -> Vec<Inst> {
    code.lines()
        .filter(|line| !line.chars().all(char::is_whitespace))
        .filter(|line| !line.contains("META")) // FIXME: process failed pattern matching
        .map(|line| Inst::parse(line).unwrap_or_else(|| panic!("Failed to parse: {line}")))
        .collect()
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Filepath to stack machine code (.sm)
    #[arg(short, long)]
    source: PathBuf,

    /// Path to output dir
    #[arg(short, long, default_value = None)]
    graphs_dir: Option<PathBuf>,

    /// Eliminate dead code
    #[arg(short, long, default_value_t = false)]
    elim_dead_code: bool,

    /// Do not generate separate store node in data graph
    #[arg(long, default_value_t = true)]
    elim_stores: bool,

    /// Delete unused stores
    #[arg(long, default_value_t = false)]
    liveliness_analysis: bool,

    /// Propagate constant
    #[arg(short, long, default_value_t = false)]
    const_prop: bool,

    /// Replace tag check on known tag with constant
    #[arg(short, long, default_value_t = false)]
    tag_check_eval: bool,

    /// Loop function body if output node in flow graph
    /// is recursive call
    #[arg(long, default_value_t = false)]
    tail_call: bool,

    /// Enable automatic inline function detection
    #[arg(long, default_value_t = false)]
    inline_strategy: bool,

    /// Remove unused declarations
    #[arg(long, default_value_t = false)]
    remove_unused_decls: bool,

    /// Replace conditional jump on constant with unconditional jump
    #[arg(short, long, default_value_t = false)]
    jump_on_const: bool,

    /// Merge blocks with unconditional jump between them
    #[arg(short, long, default_value_t = false)]
    merge_blocks: bool,

    /// Try optimize flow with `passes` iterations
    #[arg(short, long, default_value_t = 1)]
    passes: u32,

    /// Try optimize flow with `passes` iterations
    #[arg(long, default_value_t = 1)]
    unit_passes: u32,

    /// Forcefully try to inline following functions
    #[arg(long, value_delimiter = ' ', num_args = 1.., default_value = None)]
    force_inline: Option<Vec<String>>,

    /// Set all optimizations flags listed above
    #[arg(short = 'O', long, default_value_t = false)]
    optim_full: bool,
}

fn data_flags_from_args(args: &Args) -> DataGraphOptimFlags {
    DataGraphOptimFlags {
        elim_dead_code: args.elim_dead_code || args.optim_full,
        const_prop: args.const_prop || args.optim_full,
        tag_eval: args.tag_check_eval || args.optim_full,
        elim_stores: args.elim_stores || args.optim_full,
    }
}

fn flow_flags_from_args(args: &Args) -> FlowOptimFlags {
    let data_flags = data_flags_from_args(args);

    FlowOptimFlags {
        elim_dead_code: args.elim_dead_code || args.optim_full,
        jump_on_const: args.jump_on_const || args.optim_full,
        data_flags,
        merge_blocks: args.merge_blocks || args.optim_full,
        passes: args.passes,
        liveliness_analysis: args.liveliness_analysis || args.optim_full,
        tail_call: args.tail_call || args.optim_full,
    }
}

fn unit_flags_from_args(args: &Args) -> UnitOptimFlags {
    UnitOptimFlags {
        flow_optim: flow_flags_from_args(args),
        force_inline: args.force_inline.clone().unwrap_or_default(),
        passes: args.unit_passes,
        inline_strategy: args.inline_strategy || args.optim_full,
        remove_unused_decls: args.remove_unused_decls || args.optim_full,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let flags = unit_flags_from_args(&args);
    let content = read_to_string(File::open(&args.source)?)?;
    // eprintln!("Get SM code at rust side:");
    // eprintln!("{content}");
    let code = parse_stack_code(&content);
    let mut ctx = Ctx::default();
    let mut unit = Unit::analyze(&mut ctx, code.into_iter());
    unit.optimize(&mut ctx, flags);

    if let Some(out_dir) = args.graphs_dir {
        for (name, flow_graph) in unit.functions.iter() {
            let mut buffer = String::new();
            function_graph(&mut buffer, flow_graph)?;
            fs::write(out_dir.join(name).with_extension("dot"), buffer)?;
        }
    }

    let mut buffer = String::new();
    write_code(&mut buffer, &unit.compile())?;
    fs::write(args.source.with_extension("osm"), buffer)?;

    Ok(())
}
