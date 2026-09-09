DST="${1:-release}"   # default: release; pass e.g. -opt1 as first arg

rm -rf /home/bu/tmp/thoosa_build/*
cp target/x86_64-unknown-linux-gnu/$DST/*thoosa_* ~/tmp/thoosa_build/
cp target/$DST/libthoosa*.so ~/tmp/thoosa_build/
