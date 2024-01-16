use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();


    train_model(args.get(1).unwrap_or(&"../../data_local/simp_model".into()));

    println!("Ok")
}

fn train_model(save_dir:&str) {
    println!("save_dir={:?}", save_dir);

    let train_input_parameter_input_name = "training_input";
    let train_input_parameter_target_name = "training_target";
    let pred_input_parameter_name = "inputs";

    //Names of output nodes of the graph, retrieved with the saved_model_cli command
    let train_output_parameter_name = "output_0";
    let pred_output_parameter_name = "output_0";

    //Create some tensors to feed to the model for training, one as input and one as the target value
    //Note: All tensors must be declared before args!
    let input_tensor: Tensor<f32> = Tensor::new(&[1, 2]).with_values(&[1.0, 1.0]).unwrap();
    let target_tensor: Tensor<f32> = Tensor::new(&[1, 1]).with_values(&[2.0]).unwrap();


    //Create a graph
    let mut graph = Graph::new();

    //Load save model as graph
    let bundle = SavedModelBundle::load(
        &SessionOptions::new(), &["serve"], &mut graph, save_dir
    ).expect("Can't load saved model");

    let session = &bundle.session;

    let signature_train = bundle.meta_graph_def().get_signature("train").unwrap();

    //Input information
    let input_info_train = signature_train.get_input(train_input_parameter_input_name).unwrap();
    let target_info_train = signature_train.get_input(train_input_parameter_target_name).unwrap();

    //Output information
    let output_info_train = signature_train.get_output(train_output_parameter_name).unwrap();

    //Input operation
    let input_op_train = graph.operation_by_name_required(&input_info_train.name().name).unwrap();
    let target_op_train = graph.operation_by_name_required(&target_info_train.name().name).unwrap();

    //Output operation
    let output_op_train = graph.operation_by_name_required(&output_info_train.name().name).unwrap();


    let mut args = SessionRunArgs::new();

    //Feed the tensors into the graph
    args.add_feed(&input_op_train, 0, &input_tensor);
    args.add_feed(&target_op_train, 0, &target_tensor);

    //Fetch result from graph
    let mut out = args.request_fetch(&output_op_train, 0);

    //Run the session
    //kaggle_hms_harmful_brain_activity_classification
    session
        .run(&mut args)
        .expect("Error occurred during calculations");

    for device in session.device_list().unwrap() {
        println!("device: {:?}", device);
    }

    let loss: f32 = args.fetch(out).unwrap()[0];

    println!("xxx loss={:?}", loss);
}