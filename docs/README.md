# Documentation

## Building documentation
```
cd $PROJECT_DIR
rm docs/build/ -rf && make -C docs html && xdg-open docs/build/html/index.html 
```