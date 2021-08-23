echo "script is not meant to run"
exit 0

# freeze version of a dedicated package
find -name requirements.txt -exec sed -i -e 's/numpy/numpy==1.20.*/g' {} \;
