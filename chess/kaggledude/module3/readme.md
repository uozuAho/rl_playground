# todo
- copy paste code from RLC, see if u can get it working with latest py
- ABORT. figure out which python/tf/other deps are needed

# reverse engineering which python/tf/other dep versions are needed
Note 1: Python versioning has sucked for ages, I guess it's not the author's fault
Note 2: TF makes breaking changes all the time? stop that?

- I'm guessing python 3, since there's `print`
- handy list of TF versions and required python versions: https://www.tensorflow.org/install/source
- how to tell which combination is needed for this repo: https://github.com/arjangroen/RLC
    - dunno. trial and error. This _is_ the author's fault

## worflow
```sh
rm -rf .venv/ .python-version pyproject.toml uv.lock README.md hello.py
uv init -p <python version>
# uv add numpy pandas python-chess tensorflow<tf version spec> eg:
uv add numpy pandas python-chess "tensorflow>=2.0,<2.2"
# maybe uv add keras
uv add git+https://github.com/arjangroen/RLC.git
uv run demo.py
# repeat until demo works
```

## tried
- python 3.12, tf 2.18
- python 3.7, tf 2.1.1
- python 3.7, tf 1.15.3


# dependency hell, attempt 2
Idea: use latest python and libs, clone RLC repo, fix old api usages

```sh
git clone git@github.com:arjangroen/RLC.git temp
mv temp/RLC RLC
rm -rf temp
uv run demo
```
