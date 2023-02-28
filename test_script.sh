#!/bin/bash
declare LOOPS=$1
#Test serial Python code
echo -e "**************************************************"
echo -e "Test serial_python_v1.py ${LOOPS}"
echo -e "Naive histogram"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --cuda-memory-usage=true Python LSTMTrainingPetr4.py 
${LOOPS}
echo -e