.PHONY: huisman tests

clean:
	rm -rvf */*.pyc */*~ active*.csv huisman/pix/ __pycache__* tests/__pycache__* lasmap.egg-info tests/data

tests:
	pytest
	Rscript tests/test_simplex.r	

huisman: 
	python huisman/basics.py
	Rscript huisman/basics.r
