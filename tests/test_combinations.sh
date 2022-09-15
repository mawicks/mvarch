ARGS="--start-date 2010-01-01 --end-date 2019-12-31 --eval-start-date 2020-01-01 --eval-end-date 2022-06-30"
SYMBOL_ARGS="-s SPY -s QQQ -s BND"
for distribution in normal studentt
do
    for mean in zero constant arma
    do
	for univariate in arch none
	do
	    for multivariate in mvarch none
	    do
		for constraint in scalar diagonal triangular none
		do
		    if [ $univariate != none -o $multivariate != none ]
		    then
			echo
			echo Checking distribution:$distribution mean:$mean univariate:$univariate multivariate:$multivariate constraint:$constraint
			python -m mvarch.train $ARGS $SYMBOL_ARGS \
			       --distribution $distribution \
			       --mean $mean \
			       --univariate $univariate \
			       --multivariate $multivariate \
			       --constraint $constraint
		    fi
		done
	    done
	done
    done
done
