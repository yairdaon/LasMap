clean:
	rm -rvf */*.pyc */*~ active*.csv

test:
	python2.7 tests/test_helpers.py
	echo "The python script saves its results to a csv file. Then the R script compares its results to the python results."
	python tests/test_simplex.py
	Rscript tests/test_simplex.r
	python tests/test_comp.py

huisman: 
	python huisman/basics.py
	Rscript huisman/basics.r
