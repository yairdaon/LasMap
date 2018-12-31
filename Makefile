all:
	make clean
	# sudo pip install -e .
	make tests
	make huisman
	python main.py

.PHONY: huisman tests all

clean:
	rm -rvf */*.pyc */*~ active*.csv Huisman/pix/ __pycache__* tests/__pycache__* lasmap.egg-info tests/data

tests:
	pytest -x --pdb
	# pytest --pdb maxfail=3
	Rscript tests/test_simplex.r	

huisman:
	Rscript Huisman/huisman.r
	python huisman_basics.py
	Rscript huisman_basics.r

# huisman/raw_noiseless_huisman.csv: huisman/huisman.r
# 	Rscript huisman/huisman.r
