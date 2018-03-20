clean:
	rm -f *.pyc *~ active*.csv

test:
	./test_helpers.py
	echo "The python script saves its results to a csv file. Then the R script compares its results to the python results."
	./test_simplex.py
	./test_simplex.r
	./test_comp.py
