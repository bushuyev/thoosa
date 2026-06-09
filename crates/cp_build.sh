RELEASE=release #-opt1
rm -rf /home/bu/tmp/thoosa_build/*
cp target/x86_64-unknown-linux-gnu/$RELEASE/*thoosa_* ~/tmp/thoosa_build/
cp target/$RELEASE/libthoosa*.so ~/tmp/thoosa_build/
