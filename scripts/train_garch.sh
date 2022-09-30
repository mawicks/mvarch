export PYTHONPATH=$(dirname $0)
SYMBOLS="aal   aapl  amc   amzn  ba    baba  bac   bnd   c     cmcsa \
         cmg   csco  dg    dis   edv   f     fb    fdx   fxg   fxl   \
         gld   gm    gme   gps   hd    hon   hpe   ibb   intc  iyf   \
         iyr   jnj   jnk   jpm   ko    kr    lz    mgm   mmm   mro   \
         msft  mvis  nflx  nke   nvda  oxy   pep   pfe   pg    pins  \
         pton  pypl  qld   qqq   rkt   roku  rxl   sbux  snap  snow  \
         spy   sq    t     tgt   tri   trip  twtr  tyd   uber  uge   \
         ups   v     vbk   vgt   vnq   vti   vz    wfc   wmt   x     \
         xlv   xly   xmtr  xmvm  xom"

args="--distribution studentt \
      --mean zero \
      --univariate arch \
      --multivariate none "

mkdir -p models/

for symbol in $SYMBOLS
do
    # symbol=`echo $symbol | tr '[:lower:]' '[:upper:]'`
    echo "Training model for $symbol"
    python -m mvarch.train  $args $* -s $symbol -o models/${symbol}.pt
done

