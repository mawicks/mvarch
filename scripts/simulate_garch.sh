export PYTHONPATH=$(dirname $0)
mkdir -p simulations/

args="-p 90 \
      -s 2048 "

for model_file in models/*.pt
do
    symbol=`echo $model_file | sed 'sxmodels/xx' | sed 's/.pt//'`
    symbol=`echo $symbol | tr '[:lower:]' '[:upper:]'`
    echo "Simulating $symbol"
    python -m mvarch.simulate $args --output_file simulations/${symbol}.pkl $* ${model_file}
done

