How to run this model in your local?

1. Create a new environment using this command
```python3 -m venv .venv```

2. Activate your environment using this command
```source .venv/bin/activate```

3. install all requirements from the app using
```make install_req```

4. Download all required assets (it will take quite long since our model quite large)
```make download_assets```

5. run the app in local using
```streamlit run App.py```
