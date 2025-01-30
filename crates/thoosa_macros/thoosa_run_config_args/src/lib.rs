use std::fmt::Debug;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::resource::ResourcesConfig;
use thoosa_shared::val::{NamedRange, NamedVal};
use clap::{arg, command, value_parser, Arg, ArgAction, Command};

pub trait ThoosaArgsCommand {
    fn command() -> Command;
}

pub trait ThoosaArgs {
    fn args(prefix: Option<&str>) -> (Option<Vec<impl Into<Arg>>>, Option<Vec<Command>>);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it() {

    }
}

