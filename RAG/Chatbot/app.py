import os, sys
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
from functools import wraps

from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Text, Float, Enum, ForeignKey, TIMESTAMP, func, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, scoped_session

# === RAG function (tanpa CE) ===
from query_rag_mistral import get_chatbot_response_with_metrics

# === ROUGE ===
from rouge_score import rouge_scorer

from dotenv import load_dotenv; load_dotenv()

# =========================
# Flask & Database setup
# =========================
app = Flask(
    __name__,
    template_folder=os.path.join(APP_DIR, "templates"),
    static_folder=os.path.join(APP_DIR, "static"),
)
app.secret_key = "CHANGE_ME"
app.permanent_session_lifetime = timedelta(days=7)

# Sesuaikan kredensial MySQL Anda
DB_URI = "mysql+pymysql://root:@localhost/ragdb"
engine = create_engine(DB_URI, pool_pre_ping=True)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
Base = declarative_base()

# =========================
# Models
# =========================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum('user','admin'), default='user')
    created_at = Column(TIMESTAMP, server_default=func.now())

    # relasi
    queries = relationship(
        "Query",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

class Query(Base):
    __tablename__ = "queries"
    id = Column(BigInteger, primary_key=True)

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    question = Column(Text, nullable=False)
    llm_answer = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())

    user = relationship("User", back_populates="queries")

    logs = relationship(
        "RetrievalLog",
        back_populates="query",
        cascade="all, delete-orphan",
        passive_deletes=True
    )
    evaluations = relationship(
        "Evaluation",
        back_populates="query",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"
    id = Column(BigInteger, primary_key=True)
    query_id = Column(BigInteger, ForeignKey("queries.id", ondelete="CASCADE"), nullable=False)
    rank_int = Column(Integer, nullable=False)
    cosine_score = Column(Float)
    content_preview = Column(Text)
    is_context_final = Column(Boolean, default=False)

    query = relationship("Query", back_populates="logs")

class Evaluation(Base):
    __tablename__ = "evaluations"
    id = Column(BigInteger, primary_key=True)
    query_id = Column(BigInteger, ForeignKey("queries.id", ondelete="CASCADE"), nullable=False)

    reference_answer = Column(Text, nullable=False)

    # ROUGE-1
    rouge1_p  = Column(Float)
    rouge1_r  = Column(Float)
    rouge1_f1 = Column(Float)

    # ROUGE-2
    rouge2_p  = Column(Float)
    rouge2_r  = Column(Float)
    rouge2_f1 = Column(Float)

    # ROUGE-L
    rougeL_p  = Column(Float)
    rougeL_r  = Column(Float)
    rougeL_f1 = Column(Float)

    notes = Column(Text)
    evaluator_id = Column(Integer, ForeignKey("users.id"))
    evaluated_at = Column(TIMESTAMP, server_default=func.now())

    query = relationship("Query", back_populates="evaluations")

# Pastikan tabel tersedia
Base.metadata.create_all(bind=engine)

# =========================
# Helpers
# =========================
def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        db = SessionLocal()
        try:
            u = db.get(User, session["user_id"])
            if not u or u.role != "admin":
                flash("Admin only", "warning")
                return redirect(url_for("home"))
        finally:
            db.close()
        return f(*args, **kwargs)
    return wrapper

# =========================
# Routes: Auth
# =========================
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")

        if not username or not password:
            flash("Username & password wajib.", "danger")
            return redirect(url_for("register"))

        db = SessionLocal()
        try:
            if db.query(User).filter(User.username==username).first():
                flash("Username sudah dipakai.", "danger")
                return redirect(url_for("register"))
            u = User(
                username=username,
                password_hash=generate_password_hash(password),
                role='user'
            )
            db.add(u)
            db.commit()
            flash("Registrasi berhasil. Silakan login.", "success")
            return redirect(url_for("login"))
        finally:
            db.close()

    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")

        db = SessionLocal()
        try:
            u = db.query(User).filter(User.username==username).first()
            if not u or not check_password_hash(u.password_hash, password):
                flash("Username/password salah.", "danger")
                return redirect(url_for("login"))

            session["user_id"] = u.id
            session["username"] = u.username
            session["role"] = u.role

            # admin ke halaman admin, user ke home
            if u.role == "admin":
                return redirect(url_for("admin_users"))
            return redirect(url_for("home"))
        finally:
            db.close()
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# =========================
# Routes: Chat
# =========================
@app.route("/")
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.form["user_message"]

    # Panggil RAG + metrik (cosine only, tanpa CE)
    rag = get_chatbot_response_with_metrics(user_message)
    answer = rag["answer"]

    # Simpan ke MySQL
    db = SessionLocal()
    try:
        q = Query(user_id=session["user_id"], question=user_message, llm_answer=answer)
        db.add(q)
        db.flush()  # untuk dapat q.id

        # simpan semua kandidat (hanya cosine & flag chosen)
        for c in rag["candidates"]:
            log = RetrievalLog(
                query_id=q.id,
                rank_int=int(c["rank"]),
                cosine_score=float(c["cos"]) if c.get("cos") is not None else None,
                content_preview=c.get("preview"),
                is_context_final=bool(c.get("chosen"))
            )
            db.add(log)

        db.commit()
    finally:
        db.close()

    return jsonify({"response": answer})

# HISTORY SEDERHANA
@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))

    db = SessionLocal()
    try:
        uid = session["user_id"]
        qs = (db.query(Query)
                .filter(Query.user_id == uid)
                .order_by(Query.created_at.desc())
                .all())

        rows = []
        for q in qs:
            latest_eval = None
            if q.evaluations:
                # ambil yang terbaru
                latest_eval = max(q.evaluations, key=lambda e: e.evaluated_at or 0)

            rows.append({
                "id": q.id,
                "question": q.question,
                "llm_answer": q.llm_answer,
                "evaluation": None if not latest_eval else {
                    "ref":  latest_eval.reference_answer,
                    "r1_p": latest_eval.rouge1_p,  "r1_r": latest_eval.rouge1_r,  "r1_f1": latest_eval.rouge1_f1,
                    "r2_p": latest_eval.rouge2_p,  "r2_r": latest_eval.rouge2_r,  "r2_f1": latest_eval.rouge2_f1,
                    "rl_p": latest_eval.rougeL_p,  "rl_r": latest_eval.rougeL_r,  "rl_f1": latest_eval.rougeL_f1,
                }
            })

        return render_template("history.html", rows=rows)
    finally:
        db.close()


@app.post("/history/evaluate/<int:qid>")
def history_evaluate(qid):
    if "user_id" not in session:
        return redirect(url_for("login"))

    ref = (request.form.get("reference") or "").strip()
    if not ref:
        flash("Masukkan jawaban rujukan.", "warning")
        return redirect(url_for("history"))

    db = SessionLocal()
    try:
        q = db.get(Query, qid)
        if not q or q.user_id != session["user_id"]:
            flash("Tidak berhak mengevaluasi item ini.", "danger")
            return redirect(url_for("history"))

        scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
        scores = scorer.score(ref, q.llm_answer or "")
        r1, r2, rL = scores["rouge1"], scores["rouge2"], scores["rougeL"]

        db.add(Evaluation(
            query_id=q.id, reference_answer=ref,
            rouge1_p=float(r1.precision), rouge1_r=float(r1.recall), rouge1_f1=float(r1.fmeasure),
            rouge2_p=float(r2.precision), rouge2_r=float(r2.recall), rouge2_f1=float(r2.fmeasure),
            rougeL_p=float(rL.precision), rougeL_r=float(rL.recall), rougeL_f1=float(rL.fmeasure),
            evaluator_id=session["user_id"]
        ))
        db.commit()
        flash("Evaluasi ROUGE tersimpan.", "success")
    except Exception as e:
        db.rollback()
        flash(f"Gagal evaluasi: {e}", "danger")
    finally:
        db.close()
    return redirect(url_for("history"))


# ==== HAPUS SATU ====
@app.route("/history/delete/<int:qid>", methods=["POST"])
def history_delete(qid):
    if "user_id" not in session:
        return redirect(url_for("login"))
    db = SessionLocal()
    try:
        uid = session["user_id"]
        q = db.get(Query, qid)
        if not q or q.user_id != uid:
            flash("Tidak boleh menghapus item ini.", "warning")
            return redirect(url_for("history"))
        db.delete(q)
        db.commit()
        flash("Riwayat dihapus.", "success")
    except Exception as e:
        db.rollback()
        flash(f"Gagal menghapus: {e}", "danger")
    finally:
        db.close()
    return redirect(url_for("history"))

@app.route("/helper")
def helper():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("helper.html")


# =========================
# Routes: Admin
# =========================
@app.route("/admin/users")
@admin_required
def admin_users():
    db = SessionLocal()
    try:
        users = db.query(User).order_by(User.created_at.asc()).all()
        return render_template("admin_users.html", users=users)
    finally:
        db.close()

@app.route("/admin/user/<int:uid>/edit", methods=["GET","POST"])
@admin_required
def admin_user_edit(uid):
    db = SessionLocal()
    try:
        u = db.get(User, uid)
        if not u:
            flash("User tidak ditemukan.", "warning")
            return redirect(url_for("admin_users"))

        if request.method == "POST":
            action = request.form.get("action")

            # === SAVE SEMUA PERUBAHAN ===
            if action == "save_all":
                new_username = request.form.get("username", "").strip()
                new_role     = request.form.get("role", "user")
                new_pass     = request.form.get("password", "")

                if not new_username:
                    flash("Username wajib diisi.", "danger")
                    return redirect(url_for("admin_user_edit", uid=uid))

                exists = db.query(User).filter(
                    User.username == new_username,
                    User.id != u.id
                ).first()
                if exists:
                    flash("Username sudah dipakai.", "danger")
                    return redirect(url_for("admin_user_edit", uid=uid))

                u.username = new_username
                u.role     = new_role if new_role in ("user", "admin") else "user"
                if new_pass:
                    u.password_hash = generate_password_hash(new_pass)

                db.commit()
                flash("Perubahan disimpan.", "success")

                if session.get("user_id") == u.id:
                    session["username"] = u.username
                    session["role"]     = u.role

                return redirect(url_for("admin_user_edit", uid=uid))

            elif action == "update_profile":
                new_username = request.form.get("username","").strip()
                new_role     = request.form.get("role","user")

                if not new_username:
                    flash("Username wajib diisi.", "danger")
                    return redirect(url_for("admin_user_edit", uid=uid))

                exists = db.query(User).filter(
                    User.username==new_username, User.id!=u.id
                ).first()
                if exists:
                    flash("Username sudah dipakai.", "danger")
                    return redirect(url_for("admin_user_edit", uid=uid))

                u.username = new_username
                u.role     = new_role if new_role in ("user","admin") else "user"
                db.commit()
                flash("Profil user diperbarui.", "success")

                if session.get("user_id") == u.id:
                    session["username"] = u.username
                    session["role"]     = u.role

                return redirect(url_for("admin_user_edit", uid=uid))

            elif action == "update_password":
                new_pass = request.form.get("password","")
                if not new_pass:
                    flash("Password baru tidak boleh kosong.", "danger")
                    return redirect(url_for("admin_user_edit", uid=uid))
                u.password_hash = generate_password_hash(new_pass)
                db.commit()
                flash("Password user diperbarui.", "success")
                return redirect(url_for("admin_user_edit", uid=uid))

            elif action == "delete_user":
                if u.id == session.get("user_id"):
                    flash("Tidak boleh menghapus akun yang sedang login.", "warning")
                    return redirect(url_for("admin_user_edit", uid=uid))

                # hapus semua Query milik user (cascade ke logs & evaluations)
                qs = db.query(Query).filter(Query.user_id == u.id).all()
                for q in qs:
                    db.delete(q)
                db.flush()
                db.delete(u)
                db.commit()

                flash("User dan seluruh datanya dihapus.", "success")
                return redirect(url_for("admin_users"))

        return render_template("admin_user_edit.html", u=u)
    finally:
        db.close()

@app.route("/admin/users/delete_all", methods=["POST"])
@admin_required
def admin_users_delete_all():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        keep_id = session.get("user_id")
        deleted = 0
        for u in users:
            if u.id == keep_id:
                continue
            db.delete(u)
            deleted += 1
        db.commit()
        if deleted:
            flash(f"Berhasil menghapus {deleted} user (selain akun Anda).", "success")
        else:
            flash("Tidak ada user yang dihapus.", "info")
    except Exception as e:
        db.rollback()
        flash(f"Gagal menghapus semua user: {e}", "danger")
    finally:
        db.close()
    return redirect(url_for("admin_users"))

@app.route("/admin/user/<int:uid>/queries")
@admin_required
def admin_user_queries(uid):
    db = SessionLocal()
    try:
        u = db.get(User, uid)
        if not u:
            flash("User tidak ditemukan.", "warning")
            return redirect(url_for("admin_users"))
        rows = (db.query(Query)
                  .filter(Query.user_id==uid)
                  .order_by(Query.created_at.desc())
                  .all())
        return render_template("admin_user_queries.html", user=u, rows=rows)
    finally:
        db.close()

@app.route("/admin/user/<int:uid>/queries/delete_all", methods=["POST"])
@admin_required
def admin_user_queries_delete_all(uid):
    db = SessionLocal()
    try:
        u = db.get(User, uid)
        if not u:
            flash("User tidak ditemukan.", "warning")
            return redirect(url_for("admin_users"))
        qs = db.query(Query).filter(Query.user_id==uid).all()
        for q in qs:
            db.delete(q)
        db.commit()
        flash(f"Semua query milik {u.username} dihapus.", "success")
    except Exception as e:
        db.rollback()
        flash(f"Gagal hapus semua query user: {e}", "danger")
    finally:
        db.close()
    return redirect(url_for("admin_user_queries", uid=uid))

@app.route("/admin/query/<int:qid>/delete", methods=["POST"])
@admin_required
def admin_query_delete(qid):
    dbs = SessionLocal()
    uid = request.form.get("uid") or request.args.get("uid")
    try:
        q = dbs.get(Query, qid)
        if not q:
            flash("Query tidak ditemukan.", "warning")
            return redirect(url_for("admin_user_queries", uid=uid)) if uid else redirect(url_for("admin_users"))
        dbs.delete(q)
        dbs.commit()
    except Exception as e:
        dbs.rollback()
        flash(f"Gagal menghapus: {e}", "danger")
    finally:
        dbs.close()

    return redirect(url_for("admin_user_queries", uid=uid)) if uid else redirect(url_for("admin_users"))

@app.route("/admin/query/<int:qid>", methods=["GET","POST"])
@admin_required
def admin_query_detail(qid):
    db = SessionLocal()
    try:
        q = db.get(Query, qid)
        if not q:
            flash("Query tidak ditemukan.", "warning")
            return redirect(url_for("admin_users"))

        logs = (db.query(RetrievalLog)
                  .filter(RetrievalLog.query_id==qid)
                  .order_by(RetrievalLog.rank_int.asc())
                  .all())
        eval_ = (db.query(Evaluation)
                   .filter(Evaluation.query_id==qid)
                   .order_by(Evaluation.evaluated_at.desc())
                   .first())

        uid = request.args.get("uid", type=int) or (q.user_id if q else None)
        back_url = url_for("admin_user_queries", uid=uid) if uid else url_for("admin_users")

        if request.method == "POST":
            ref = request.form.get("reference","").strip()
            if not ref:
                flash("Masukkan jawaban rujukan.", "danger")
                return redirect(url_for("admin_query_detail", qid=qid, uid=uid))

            scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
            scores = scorer.score(ref, q.llm_answer or "")

            r1, r2, rL = scores["rouge1"], scores["rouge2"], scores["rougeL"]
            ev = Evaluation(
                query_id=q.id, reference_answer=ref,
                rouge1_p=float(r1.precision), rouge1_r=float(r1.recall), rouge1_f1=float(r1.fmeasure),
                rouge2_p=float(r2.precision), rouge2_r=float(r2.recall), rouge2_f1=float(r2.fmeasure),
                rougeL_p=float(rL.precision), rougeL_r=float(rL.recall), rougeL_f1=float(rL.fmeasure),
                notes=None, evaluator_id=session["user_id"]
            )
            db.add(ev); db.commit()
            flash("Evaluasi tersimpan.", "success")
            return redirect(url_for("admin_query_detail", qid=qid, uid=uid))

        return render_template("admin_query_detail.html", q=q, logs=logs, eval_=eval_, back_url=back_url)
    finally:
        db.close()

@app.route("/whoami")
def whoami():
    dbs = SessionLocal()
    try:
        uid = session.get("user_id")
        if not uid:
            return "Not logged in"
        u = dbs.get(User, uid)
        return f"user_id={uid}, username={getattr(u,'username',None)}, role={getattr(u,'role',None)}"
    finally:
        dbs.close()

if __name__ == "__main__":
    app.run(debug=True)
