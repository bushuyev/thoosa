
enum Runs {
    ImportResources,
    FindMotifs,
    SelectMotifs,
    TrainMotifsNet,
    TrainLabelsNet,
    TrainFinalNet,
}

trait  Run {

}

struct RunsRecorder {

}

struct WsConfig {

}

struct RunConfig {

}

struct Input {

}



fn run_it(
    ws_config:WsConfig,
    run_config:RunConfig,
    input:Input,
){

}


#[cfg(test)]
pub mod tests {
    use thoosa_shared::init_test_logging;

    #[ctor::ctor]
    fn init() {
        init_test_logging();
    }


    #[test]
    fn test_0(){
        println!("Ok");


    }
}