# Documentation

## Building documentation
```
cd $PROJECT_DIR
rm docs/build/ docs/source/dev/_autosummary/ -rf && make -C docs html && xdg-open docs/build/html/index.html 
```