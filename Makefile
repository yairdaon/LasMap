.PHONY: huisman tests

clean:
	rm -rvf */*.pyc */*~ active*.csv Huisman/pix/ __pycache__* tests/__pycache__* lasmap.egg-info tests/data

tests:
	pytest -x --pdb
	# pytest --pdb maxfail=3
	Rscript tests/test_simplex.r	

huisman:
	# huisman/raw_noiseless_huisman.csv
	python huisman_basics.py
	Rscript huisman_basics.r

# huisman/raw_noiseless_huisman.csv: huisman/huisman.r
# 	Rscript huisman/huisman.r
