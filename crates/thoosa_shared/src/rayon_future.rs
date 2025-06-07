use futures::channel::oneshot;
use futures_util::task::{waker, ArcWake};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use futures::channel::oneshot::Canceled;
use log::error;

pub trait ScopeFutureExt<'scope> {
    fn spawn_future<Fut>(&self, future: Fut) -> RayonFuture<Fut::Output>
    where
        Fut: Future + Send + 'static,
        Fut::Output: Send + 'static;

}

struct Shared<T> {
    fut: Option<Pin<Box<dyn Future<Output = T> + Send + 'static>>>,
    tx: Option<oneshot::Sender<T>>, // `None` after result sent.
}

fn poll_once<T>( shared: &Arc<Mutex<Shared<T>>>)
where
    T: Send + 'static,
{
    let mut guard = shared.lock().unwrap();
    let fut = match guard.fut.take() {
        Some(f) => f,
        None => return, // already completed
    };
    drop(guard);


    let waker = waker(Arc::new(Requeue {
        shared: shared.clone(),
    }));
    let mut cx = Context::from_waker(&waker);
    let mut fut = fut;

    match fut.as_mut().poll(&mut cx) {
        Poll::Ready(output) => {
            if let Some(tx) = shared.lock().unwrap().tx.take() {
                let _ = tx.send(output);
            }
        }
        Poll::Pending => {
            shared.lock().unwrap().fut = Some(fut); 
        }
    }
}


struct Requeue<T> {
    shared: Arc<Mutex<Shared<T>>>,
}

impl<T> ArcWake for Requeue<T>
where
    T: Send + 'static,
{
    fn wake_by_ref(arc_self: &Arc<Self>) {
        let shared = arc_self.shared.clone();
        // println!("wake_by_ref: thread::current().name()={:?}", std::thread::current().name());
        rayon::spawn(move || poll_once(&shared));
    }
}

impl<'scope> ScopeFutureExt<'scope> for rayon::Scope<'scope> {
    fn spawn_future<Fut>(&self, future: Fut) -> RayonFuture<Fut::Output>
    where
        Fut: Future + Send + 'static,
        Fut::Output: Send + 'static,
    {
        let (tx, rx) = oneshot::channel::<Fut::Output>();


        let shared = Arc::new(Mutex::new(Shared {
            fut: Some(Box::pin(future)),
            tx: Some(tx),
        }));


        self.spawn(move |_| poll_once(&shared));

        RayonFuture { rx }
    }


}


#[derive(Debug)]
pub struct RayonFuture<T> {
    rx: oneshot::Receiver<T>,
}

impl<T> RayonFuture<T> {

    pub fn install_future<Fut>(future: Fut) -> RayonFuture<Fut::Output>
    where
        Fut: Future + Send + 'static,
        Fut::Output: Send + 'static,
    {
        let (tx, rx) = oneshot::channel::<Fut::Output>();


        let shared = Arc::new(Mutex::new(Shared {
            fut: Some(Box::pin(future)),
            tx: Some(tx),
        }));

        rayon::spawn(move || poll_once(&shared));

        RayonFuture { rx }
    }
}

impl<T> Future for RayonFuture<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.rx).poll(cx).map(|r| {
            match r {
                Ok(v) => v,
                Err(e) => {
                    error!("error in future: {:?}", e);
                    panic!("error in future: {:?}", e);
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;
    use futures::executor::block_on;
    use futures_util::future::join_all;
    use parking_lot::RwLock;
    use rayon::prelude::IntoParallelIterator;
    use rayon::iter::ParallelIterator;
    use crate::applicables_runner::Statistics;
    use crate::rayon_future::{RayonFuture, ScopeFutureExt};

    #[test]
    fn test_0(){

        let pool_0 = rayon::ThreadPoolBuilder::new().num_threads(4).spawn_handler(|thread_builder| {
            let b = std::thread::Builder::new();
            b.name(format!("{}/{:?}", "pool_0", thread_builder.index()))
                .spawn(|| thread_builder.run())?;

            Ok(())
        })
            .num_threads(2)
            .build().unwrap();

        pool_0.scope(|scope|{
            for i in 0..10 {
                scope.spawn(move |_|{
                    println!("preload {i}");
                });
            }

            let sum = block_on(join_all((0..10).into_iter().map(|i| {
                scope.spawn_future( async move {
                    let a = async {
                        // thread::sleep(Duration::from_millis(100));
                        println!("hello 0 0 {i} thread={:?}", thread::current().name());
                        i*10
                    }.await;

                    let b = async {
                        // thread::sleep(Duration::from_millis(100));
                        println!("hello 0 1 {i} thread={:?}", thread::current().name());
                        i
                    }.await;
                    println!("hello 0 {i} thread={:?}", thread::current().name());

                    a + b
                })

            }))).iter().sum::<i32>();

            println!("sum {sum}");//450+45

            assert_eq!(sum, 495);

        });

    }

    #[test]
    fn test_1(){
        let main_pool = rayon::ThreadPoolBuilder::new()
            .spawn_handler(|thread_builder| {
                let b = std::thread::Builder::new();
                b.name(format!("{}/{:?}", "main_pool", thread_builder.index()))
                    .spawn(|| thread_builder.run())?;

                Ok(())
            })
            .num_threads(10)
            .build()
            .expect("main pool");




        let t_0 = Instant::now();
        let statistics = Arc::new(Statistics::new());

        main_pool.scope(|scope| {
            let _ = (0..10).into_par_iter().map(|_| {
                let load_time = Arc::clone(&statistics);
                block_on(async move {//done in 5567 load time=54473
                    crate::applicables_runner::dummy_load(&load_time, &Arc::new("key".to_string())).await;
                });
            }).collect::<Vec<_>>();
        });
        println!("done in {:?} load time 0={}", t_0.elapsed().as_millis(), statistics.load_time());

        let t_1 = Instant::now();
        let load_time_1 = Arc::new(Statistics::new());

        main_pool.scope(|scope| {
            let _ = block_on(join_all((0..10)./*into_par_iter().*/map(|_| {
                let load_time = Arc::clone(&load_time_1);
                scope.spawn_future(async move {
                    crate::applicables_runner::dummy_load(&load_time, &Arc::new("key".to_string())).await;
                })
            }).collect::<Vec<_>>()));
        });


        println!("done in {:?} load time 1={}", t_1.elapsed().as_millis(),  load_time_1.load_time());


        let t_2 = Instant::now();
        let load_time_2 = Arc::new(Statistics::new());
        main_pool.install(|| {
            let _ = block_on(join_all((0..10)./*into_par_iter().*/map(|_| {
                let load_time = Arc::clone(&load_time_2);
                RayonFuture::<()>::install_future(async move {
                    crate::applicables_runner::dummy_load(&load_time, &Arc::new("key".to_string())).await;
                })
            }).collect::<Vec<_>>()));
        });


        println!("done in {:?} load time 2={}", t_2.elapsed().as_millis(),  load_time_2.load_time());
    }

}
