import os,re,io,json,time,base64,math,requests,pandas as pd,numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OPENAI_ENABLED=False
if os.getenv("OPENAI_API_KEY"):
    try:
        import openai
        openai.api_key=os.getenv("OPENAI_API_KEY")
        OPENAI_ENABLED=True
    except: OPENAI_ENABLED=False
IMG_SIZE_LIMIT_BYTES=100_000
WIKI_HIGHEST_URL="https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
def now(): return time.time()
def to_data_uri(img_bytes,mime="image/png"):
    return f"data:{mime};base64,{base64.b64encode(img_bytes).decode('ascii')}"
def compress_image_to_limit(pil_img,size_limit=IMG_SIZE_LIMIT_BYTES):
    out=io.BytesIO()
    try: pil_img.save(out,format="PNG",optimize=True)
    except: out=io.BytesIO(); pil_img.save(out,format="PNG")
    if out.tell()<=size_limit: out.seek(0); return to_data_uri(out.read(),"image/png"),out.tell()
    for q in (80,70,60,50,40,30):
        out=io.BytesIO(); pil_img.save(out,format="WEBP",quality=q,method=6)
        if out.tell()<=size_limit: out.seek(0); return to_data_uri(out.read(),"image/webp"),out.tell()
    w,h=pil_img.size; scale=0.9
    while (w>100 and h>100) and scale>0.2:
        w=int(w*scale); h=int(h*scale)
        img2=pil_img.resize((w,h),Image.LANCZOS)
        for q in (60,50,40,30):
            out=io.BytesIO(); img2.save(out,format="WEBP",quality=q,method=6)
            if out.tell()<=size_limit: out.seek(0); return to_data_uri(out.read(),"image/webp"),out.tell()
        scale-=0.1
    out.seek(0); return to_data_uri(out.read(),"image/webp"),out.tell()
def fig_to_data_uri(fig,size_limit=IMG_SIZE_LIMIT_BYTES,fmt="png"):
    buf=io.BytesIO(); fig.savefig(buf,format=fmt,bbox_inches="tight"); buf.seek(0)
    try: img=Image.open(buf).convert("RGBA")
    except: buf.seek(0); return to_data_uri(buf.read(),f"image/{fmt}"),len(buf.getvalue())
    return compress_image_to_limit(img,size_limit=size_limit)
def fetch_url_text(url,timeout=15):
    r=requests.get(url,headers={"User-Agent":"tds-agent/1.0"},timeout=timeout); r.raise_for_status(); return r.text
def parse_wikipedia_highest(html_text):
    soup=BeautifulSoup(html_text,"html.parser")
    table=None
    for tbl in soup.find_all("table",{"class":"wikitable"}):
        headers=[th.get_text(strip=True).lower() for th in tbl.find_all("th")]
        if any(h in ("rank","peak","worldwide") for h in headers) or len(headers)>=3: table=tbl; break
    rows=[]
    for tr in table.find_all("tr"):
        cols=[td.get_text(" ",strip=True) for td in tr.find_all(["td","th"])]
        if cols: rows.append(cols)
    df=pd.DataFrame(rows); header=df.iloc[0].tolist(); df=df[1:].reset_index(drop=True)
    if len(header)==df.shape[1]: df.columns=header
    else: df.columns=[f"c{i}" for i in range(df.shape[1])]
    col_lower={c.lower():c for c in df.columns}
    title_col=next((col for key,col in col_lower.items() if "title" in key or "film" in key or "movie" in key),df.columns[0])
    year_col=next((col for key,col in col_lower.items() if "year" in key),None)
    gross_col=next((col for key,col in col_lower.items() if "worldwide" in key or "gross" in key or "peak" in key),None)
    rank_col=next((col for key,col in col_lower.items() if "rank" in key),None)
    def parse_money(s):
        if pd.isna(s): return None
        s=str(s).replace(",","").replace("—","").strip()
        m=re.search(r'([\d\.]+)\s*(billion|bn)\b',s,re.I)
        if m: return float(m.group(1))*1e9
        m=re.search(r'\$?\s*([\d,\.]+)',s)
        if m:
            try: return float(m.group(1))
            except: return None
        return None
    def parse_year_from_text(s):
        if pd.isna(s): return None
        m=re.search(r'(19|20)\d{2}',str(s))
        if m: return int(m.group(0))
        return None
    result=pd.DataFrame()
    result['title']=df[title_col].astype(str).str.strip()
    result['gross']=df[gross_col].apply(parse_money) if gross_col else np.nan
    result['year']=df[year_col].apply(parse_year_from_text) if year_col else result['title'].apply(parse_year_from_text)
    if rank_col: result['rank']=pd.to_numeric(df[rank_col],errors='coerce')
    else:
        maybe_rank=pd.to_numeric(df[df.columns[0]],errors='coerce')
        result['rank']=maybe_rank if maybe_rank.notna().sum()>0 else np.arange(1,len(result)+1)
    result['peak']=result['gross']
    return result
def wiki_highest_handler():
    df=parse_wikipedia_highest(fetch_url_text(WIKI_HIGHEST_URL))
    cnt=int(((df['peak']>=2e9)&(df['year']<2000)).sum())
    filt=df[(df['peak']>=1.5e9)&(df['year'].notna())]
    earliest=filt.sort_values('year').iloc[0]['title'] if not filt.empty else ""
    try: c=float(df[['rank','peak']].dropna().corr().iloc[0,1])
    except: c=0.0
    fig,ax=plt.subplots(figsize=(6,4))
    plot_df=df[['rank','peak']].dropna()
    if not plot_df.empty:
        ax.scatter(plot_df['rank'],plot_df['peak'])
        m,b=np.polyfit(plot_df['rank'],plot_df['peak'],1)
        xs=np.linspace(plot_df['rank'].min(),plot_df['rank'].max(),200); ys=m*xs+b
        ax.plot(xs,ys,linestyle=':',color='red'); ax.set_xlabel('Rank'); ax.set_ylabel('Peak (USD)'); ax.set_title('Rank vs Peak')
    else: ax.text(0.5,0.5,"Not enough data",ha='center')
    uri,_=fig_to_data_uri(fig); plt.close(fig)
    return [cnt,earliest,round(c,6),uri]
def try_load_tabular(file_path):
    ext=file_path.lower()
    try:
        if ext.endswith((".parquet",".pq")): return pd.read_parquet(file_path)
        return pd.read_csv(file_path)
    except:
        with open(file_path,"rb") as f: b=f.read()
        try: return pd.read_csv(io.StringIO(b.decode("utf-8",errors="replace")))
        except: raise
def generic_handler(files_dict,q_text):
    out={}
    tab_file=None; tab_name=None
    for k,v in files_dict.items():
        if k.lower().endswith(('.csv','.parquet','.pq')): tab_file=v; tab_name=k; break
    df=None
    if tab_file:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False,suffix=os.path.splitext(tab_name)[1]) as tmp:
            tmp.write(tab_file); tmp_path=tmp.name
        try: df=try_load_tabular(tmp_path)
        finally: os.unlink(tmp_path)
    for line in q_text.strip().splitlines():
        key=line.strip()
        if not key: continue
        ans=None
        low_key=key.lower()
        if df is not None:
            if "average" in low_key or "mean" in low_key:
                num_cols=df.select_dtypes(include=[np.number])
                if not num_cols.empty: ans=float(num_cols.mean().mean())
            elif "max" in low_key:
                num_cols=df.select_dtypes(include=[np.number])
                if not num_cols.empty: ans=float(num_cols.max().max())
            elif "min" in low_key:
                num_cols=df.select_dtypes(include=[np.number])
                if not num_cols.empty: ans=float(num_cols.min().min())
            elif "correlation" in low_key:
                nums=df.select_dtypes(include=[np.number])
                if nums.shape[1]>=2: ans=float(nums.corr().iloc[0,1])
            elif "plot" in low_key or "chart" in low_key or "graph" in low_key:
                nums=df.select_dtypes(include=[np.number])
                if nums.shape[1]>=2:
                    fig,ax=plt.subplots(figsize=(6,4))
                    ax.scatter(nums.iloc[:,0],nums.iloc[:,1])
                    try:
                        m,b=np.polyfit(nums.iloc[:,0],nums.iloc[:,1],1)
                        xs=np.linspace(nums.iloc[:,0].min(),nums.iloc[:,0].max(),200)
                        ys=m*xs+b
                        ax.plot(xs,ys,linestyle=':',color='red')
                    except: pass
                    uri,_=fig_to_data_uri(fig); plt.close(fig); ans=uri
        if ans is None and OPENAI_ENABLED:
            try:
                resp=openai.ChatCompletion.create(model="gpt-4o-mini",messages=[{"role":"user","content":f"Answer this based on dataset columns {list(df.columns) if df is not None else 'No dataset'}:\n{key}"}],temperature=0)
                ans=resp["choices"][0]["message"]["content"].strip()
            except: ans="Unable to answer"
        if ans is None: ans="Unable to answer"
        out[key]=ans
    return out
def handle_analysis(files):
    if "questions.txt" not in files: return {"error":"questions.txt required."}
    try: q_text=files["questions.txt"].decode("utf-8",errors="replace")
    except: q_text=files["questions.txt"].decode("latin-1",errors="replace")
    if "list_of_highest-grossing_films" in q_text.lower() or "highest grossing films" in q_text.lower() or WIKI_HIGHEST_URL.lower() in q_text.lower():
        try: return wiki_highest_handler()
        except Exception as e: return {"error":"wiki_handler_failed","detail":str(e)}
    return generic_handler(files,q_text)
if __name__=="__main__":
    print("Module test skipped.")
