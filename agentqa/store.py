"""Small transactional JSON document store, with immutable audit entries."""
import json
import sqlite3
import threading
from pathlib import Path
from .models import now, uid

class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.connect() as c:
            c.executescript('''PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS documents(kind TEXT, id TEXT, body TEXT NOT NULL, PRIMARY KEY(kind,id));
            CREATE TABLE IF NOT EXISTS audit(id TEXT PRIMARY KEY, at TEXT, actor TEXT, action TEXT, body TEXT);
            CREATE TABLE IF NOT EXISTS claims(plan_id TEXT PRIMARY KEY, run_id TEXT, at TEXT);
            ''')
    def connect(self):
        c=sqlite3.connect(self.path, timeout=30)
        c.row_factory=sqlite3.Row
        return c
    def put(self, kind, obj):
        data=obj.model_dump(mode='json',by_alias=True) if hasattr(obj,'model_dump') else obj
        with self.lock, self.connect() as c:
            c.execute('INSERT INTO documents VALUES(?,?,?) ON CONFLICT(kind,id) DO UPDATE SET body=excluded.body',(kind,data['id'],json.dumps(data)))
        return data
    def get(self, kind, id):
        with self.connect() as c: row=c.execute('SELECT body FROM documents WHERE kind=? AND id=?',(kind,id)).fetchone()
        if not row: raise KeyError(f'{kind} not found')
        return json.loads(row['body'])
    def all(self, kind):
        with self.connect() as c: rows=c.execute('SELECT body FROM documents WHERE kind=? ORDER BY rowid DESC',(kind,)).fetchall()
        return [json.loads(r['body']) for r in rows]
    def audit(self, actor, action, body):
        with self.lock, self.connect() as c:
            c.execute('INSERT INTO audit VALUES(?,?,?,?,?)',(uid(),now(),actor,action,json.dumps(body)))
    def audits(self):
        with self.connect() as c: return [dict(r) | {'body':json.loads(r['body'])} for r in c.execute('SELECT * FROM audit ORDER BY at DESC')]
    def claim(self, plan_id, run_id):
        with self.lock, self.connect() as c:
            try: c.execute('INSERT INTO claims VALUES(?,?,?)',(plan_id,run_id,now()))
            except sqlite3.IntegrityError: raise ValueError('Plan already consumed; create a fresh plan')
