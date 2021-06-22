from setuptools import setup

with open("README.md", "r", encoding="utf8") as fh:
	long_description = fh.read()

setup(
	name= "jostar",
	version="0.0.1",
	description=long_description,
	long_description_content_type = "text/markdown",
	author="Amirhossein Hassanzadeh",
	url = "",
	author_email = "ah7557@rit.edu" ,
	packages_dir={"":"jostar"},
	keywords={"optimization", "feature-selection","genetic algorithm", "particle swarm", "ant colony", "metaheuristics", "differential evolution", "sequential search", "multi-objective"},
	classifiers=['Programming Language :: Python :: 3.6',
				"License :: OSI Approved :: MIT License",
				"Operating System :: OS Independent"],
	install_requires=["sklearn","matplotlib","pathos","scipy","seaborn"],
	extras_require={"dev":["pytest>=3.6",],}
	)
	

