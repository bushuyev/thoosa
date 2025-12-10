use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::traits::WithId;

#[derive(Debug)]
struct TrainNetInput {

}

impl Input for TrainNetInput {

}

#[derive(Debug)]
struct TrainNetOutput {

}

impl WithId for TrainNetOutput {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl Output for TrainNetOutput {

}


#[derive(Debug)]
pub struct TrainNet {

}

impl TrainNet {

}



#[cfg(test)]
mod tests {


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_0() {

    }

}