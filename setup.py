from setuptools import find_packages, setup
import os
base_dir = os.path.dirname(__file__)

with open(os.path.join(base_dir, "README.md")) as f:
    long_description = f.read()

    
setup(
	name= "jostar",
	version="0.0.4",
	description="Feature Selection Module for Data Sciences in Python",
	long_description=long_description,
	long_description_content_type = "text/markdown",
	author="Amirhossein Hassanzadeh",
	url = "",
	author_email = "ah7557@rit.edu" ,
	packages=find_packages(),
	keywords={"optimization", "feature-selection","genetic algorithm", "particle swarm", "ant colony", "metaheuristics", "differential evolution", "sequential search", "multi-objective"},
	classifiers=['Programming Language :: Python :: 3.6',
				"License :: OSI Approved :: MIT License",
				"Operating System :: OS Independent"],
	install_requires=["sklearn","matplotlib","pathos","scipy","seaborn"],
	extras_require={"dev":["pytest>=3.6",],}
	)
	

