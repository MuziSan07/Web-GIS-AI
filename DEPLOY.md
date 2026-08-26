# Putting the demo online

The Streamlit app in `streamlit_app.py` is a second front end over the same
engine as the web application. It exists so a client can be sent a link.

## What you need

- A GitHub repository containing this project
- A Streamlit Community Cloud account, at share.streamlit.io
- A Groq API key, from console.groq.com

## Steps

1. Push the project to GitHub. Do not push `.env` or `.streamlit/secrets.toml`.
   Both are already in `.gitignore`. Check with `git status` before pushing.

2. On Streamlit Community Cloud, create a new app pointing at the repository,
   with `streamlit_app.py` as the main file.

3. Open the app settings, go to Secrets, and paste:

   ```
   GROQ_API_KEY = "your-key-here"
   PLANENG_LLM_PROVIDER = "groq"
   ```

   Streamlit puts these into the environment, which is where the engine looks
   for them. The key never appears in the repository or in the interface.

4. Deploy. The first build takes several minutes because geopandas, rasterio
   and the export writers are large.

## What it costs to run

Measured on this project, a 600 hectare plan for 25,000 people:

| Grid | Time | Peak memory |
|---|---|---|
| 25 m | about 15 seconds | 145 MB |
| 20 m | about 20 seconds | 144 MB |

Community Cloud gives one CPU and a gigabyte of memory, so the plan itself is
comfortable. The finest 10 m grid is deliberately not offered in the demo.

Reading a brief costs roughly 1,500 tokens. The Groq free tier allows 8,000
tokens a minute, so a single user is fine and a burst of five briefs in a row
will be rate limited. The engine waits the limit out rather than failing, and
falls back to its rule based reader if the wait would be long.

## Things worth knowing before you send the link

**Community Cloud requires a public repository on the free tier.** Deploying
from a private repository needs a paid plan. If the source has to stay closed,
host the app somewhere else instead: it is an ordinary Python process, so any
container host will run `streamlit run streamlit_app.py`.

**The app sleeps when idle** and takes a while to wake. If the client is
opening the link at a scheduled time, open it yourself a few minutes before.

**There are no user accounts.** Anyone with the link can use it, and everyone
shares one container, so two people generating plans at once will both wait.
That is fine for a demo and is not a product.

**The plans are marked not for statutory use** while the ruleset is
placeholder. Leave that in place until a registered planner has confirmed the
standards.

## Running it locally

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Put the key in a `.env` file beside this one:

```
GROQ_API_KEY=your-key-here
PLANENG_LLM_PROVIDER=groq
```

With no key and no local Ollama, the app still runs. The brief is read by the
rule based reader instead, and the page says so.
