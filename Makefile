clean:
	rm -f *.pyc *~ active*.csv

test:
	./test_simplex.r
	./test_simplex.py
