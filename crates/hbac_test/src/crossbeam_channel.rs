use std::collections::HashMap;
use std::future::Future;
use std::ops::DerefMut;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;
use rayon::{in_place_scope, scope};
use rayon::prelude::*;
use rayon::iter::ParallelIterator;
use crossbeam_utils::sync::WaitGroup;

async fn async_load()->usize {
    100
}

trait Loader {
    fn load(&mut self, c:usize) -> usize;
}

struct TestLoader {
    cache_mx: Arc<Mutex<HashMap<usize, bool>>>
    // cache: HashMap<usize, bool>
}

struct LoadFuture {

}

impl LoadFuture {
    pub fn new() -> Self {
        Self {}
    }
}

impl Future for LoadFuture {
    type Output = bool;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {

        Poll::Ready(true)
    }
}

impl TestLoader {
    pub fn new() -> Self {
        Self {
            cache_mx:Arc::new(Mutex::new(HashMap::new()))
            }
    }

    fn do_load(cache: &mut HashMap<usize, bool>, c:usize) -> impl Future{
        println!("1. do_load:  c={}", c);

        std::thread::sleep(Duration::new(1, 0));
        cache.insert(c, true);

        println!("2. do_load:  c={}", c);

        LoadFuture::new()
    }

    fn load_bg(&mut self, c: usize) {
        println!("load_bg:  c={}", c);

        let c_mutex = Arc::clone(&self.cache_mx);

        rayon::spawn(move || {
            //load c + 1
            println!("load_bg: in pool c={}", c);
            if let Ok(mut cache) = c_mutex.lock() {
                Self::do_load(cache.deref_mut(), c);
            }
        })
    }
}

impl Loader for TestLoader {

    fn load(&mut self, c:usize) -> usize {


        let wg = WaitGroup::new();


        let mut result_a: Option<usize> = None;
        let wg_1 = wg.clone();

        in_place_scope(|s| async move {
            // s.spawn(|_|async move{
            //     println!("in_place_scope spawn");
            // });
            println!("in_place_scope");
            // Self::do_load(&mut HashMap::new(), 1).await;
            result_a.replace(10);

            drop(wg_1);
        });

        wg.wait();

        // rayon::spawn(move ||  {


        // });


        println!("result_a={:?}", result_a);

        10

        // let arc = Arc::clone(&self.cache_mx);
        // let cache_lock = arc.lock();
        //
        // if let Ok(mut cache) = cache_lock {
        //
        //     return if cache.contains_key(&c){
        //         c
        //     } else {
        //
        //         self.load_bg(c+1);
        //
        //
        //
        //
        //         c*10
        //     };
        // }
        //
        // panic!("Can't lock cache");

    }
}

struct Counter {
    count: usize,
    count_to:usize,
    loader: Box<dyn Loader + Send>
}

impl Counter {
    fn new(count_to:usize) -> Counter {
        Counter {
            count: 0,
            count_to,
            loader: Box::new(TestLoader::new())
        }
    }
}

impl Iterator for Counter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {


        if self.count < self.count_to {
            self.count+=1;
            Some(self.loader.load(self.count))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;
    use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelBridge};
    use crate::crossbeam_channel::Counter;
    use rayon::iter::ParallelIterator;
    use crossbeam::scope;
    use rayon::in_place_scope;
    use crossbeam_channel::unbounded;
    use crossbeam_utils::Backoff;
    use crossbeam_utils::sync::WaitGroup;
    use futures::executor::block_on;
    use futures::SinkExt;


    #[test]
    pub fn test_it(){
        rayon::ThreadPoolBuilder::new().num_threads(10).build_global().unwrap();

        Counter::new(10).par_bridge().for_each(|c| {
            println!("c={:?}", c)
        });
    }

    async fn some(i:i32)->i32{
        i * 10
    }

    #[test]
    pub fn test_test2() {

        // in_place_scope(move |s| {
        // scope(move |s| {
        //
        //     s.spawn(move |_|{
        let (up_s_0, up_r_0) = unbounded::<i32>();
        let (down_s_0,down_r_0) = unbounded::<i32>();
        thread::spawn(move ||{
            // println!("a={}", a);
            // thread::sleep(Duration::new(10, 0));
            let backoff = Backoff::new();
            loop {

                let x = up_r_0.recv().unwrap();
                // println!("thread got x={}", x);

                thread::sleep(Duration::new(10, 0));
                down_s_0.send(x * 1_000_000).unwrap();

                backoff.snooze();
            }
        });



        let (up_s, up_r) = unbounded::<i32>();
        let (down_s,down_r) = unbounded::<i32>();
        thread::spawn(move ||{
            let backoff = Backoff::new();
            loop {
                if let Ok(a) = up_r.try_recv() {
                    if a == 7 {
                        up_s_0.send(a);

                    } else {
                        down_s.send(
                            a * 10
                        ).unwrap();
                    }
                }


                if let Ok(r) = down_r_0.try_recv() {
                    down_s.send(
                        r
                    ).unwrap();
                }

                backoff.snooze();

                // c +=1;
                // if c > 2 {
                //     break;
                // }
                //     thread::park();
                // } else {
                //     backoff.spin();
                // }
                // thread::park();

            }

        });
        // });

        // let wg = WaitGroup::new();

        rayon::ThreadPoolBuilder::new().num_threads(10).build_global().unwrap();

        let x:i32 = (0..100).into_par_iter().map(|i|  {
            println!("1. i={}", i);

            up_s.send(i).unwrap();

            let x = down_r.recv().unwrap();

            println!("2. i={}", i);

            x
        }).sum();


        println!("x={:?}", x);
    }

    #[test]
    pub fn test_th(){
        let (up_s, up_r) = unbounded::<&str>();
        let (down_s,down_r) = unbounded::<&str>();

        thread::spawn(move ||{
            let backoff = Backoff::new();
            loop {
                let msg = up_r.recv().unwrap();
                println!("top msg={}", msg);
                down_s.send(
                    msg
                ).unwrap();

                backoff.snooze();
            }
        });

        thread::sleep(Duration::new(1, 0));

        let up_s_0 = Arc::new(Mutex::new(up_s));
        let down_r_0 = Arc::new(Mutex::new(down_r));


        let up_s_1 = Arc::clone(&up_s_0);
        let down_r_1 = Arc::clone(&down_r_0);

        thread::spawn(move ||{
            println!("thread 1");

            for i in 0..3 {
                if let Ok(up_lock) = up_s_1.lock(){
                    up_lock.send("from 1".into()).unwrap()
                }

            }

            let backoff = Backoff::new();
            loop {

                if let Ok(down_lock) = down_r_1.lock(){
                    if let Ok(msg) = down_lock.try_recv() {
                        println!("thread 1 got msg: {}", msg);
                    }
                }
                backoff.snooze();

            }
        });


        let up_s_2 = Arc::clone(&up_s_0);
        let down_r_2 = Arc::clone(&down_r_0);

        thread::spawn(move ||{
            println!("thread 2");

            for i in 0..3 {
                if let Ok(up_lock) = up_s_2.lock(){
                    up_lock.send("from 2".into()).unwrap()
                }

            }

            let backoff = Backoff::new();
            loop {

                if let Ok(down_lock) = down_r_2.lock(){
                    if let Ok(msg) = down_lock.try_recv() {
                        println!("thread 2 got msg: {}", msg);
                    }
                }
                backoff.snooze();

            }
        });

        thread::sleep(Duration::new(10, 0));
        println!("ok");
    }
}