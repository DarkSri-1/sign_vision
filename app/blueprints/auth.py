from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from urllib.parse import urlparse, urljoin as url_join
from app import db
from app.models import User
from app.forms import SignupForm, LoginForm, ForgotPasswordForm, ResetPasswordForm

bp = Blueprint("auth", __name__)


def _safe_redirect(target):
    if not target:
        return None
    ref_url = urlparse(request.host_url)
    test_url = urlparse(url_join(request.host_url, target))
    if test_url.scheme in ("http", "https") and ref_url.netloc == test_url.netloc:
        return target
    return None


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    form = SignupForm()
    if form.validate_on_submit():
        if User.query.filter_by(email=form.email.data.strip().lower()).first():
            flash("Email already exists.", "danger")
            return render_template("auth/register.html", form=form)
        if User.query.filter_by(username=form.username.data.strip()).first():
            flash("Username already taken.", "danger")
            return render_template("auth/register.html", form=form)
        u = User(
            full_name=form.full_name.data.strip(),
            email=form.email.data.strip().lower(),
            username=form.username.data.strip(),
        )
        u.set_password(form.password.data)
        db.session.add(u)
        db.session.commit()
        flash("Account created. You can sign in now.", "success")
        return redirect(url_for("auth.login"))
    return render_template("auth/register.html", form=form)


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    form = LoginForm()
    if form.validate_on_submit():
        ident = form.username_or_email.data.strip()
        user = User.query.filter(
            (User.username == ident) | (User.email == ident.lower())
        ).first()
        if user is None or not user.check_password(form.password.data):
            flash("Invalid username/email or password.", "danger")
            return render_template("auth/login.html", form=form)
        if not user.is_active:
            flash("Your account is deactivated. Contact administrator.", "warning")
            return render_template("auth/login.html", form=form)
        login_user(user, remember=form.remember.data)
        from datetime import datetime

        user.last_login = datetime.utcnow()
        user.last_activity = datetime.utcnow()
        db.session.commit()
        next_page = _safe_redirect(request.args.get("next")) or url_for("main.dashboard")
        flash("Signed in successfully.", "success")
        return redirect(next_page)
    return render_template("auth/login.html", form=form)


@bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been signed out.", "info")
    return redirect(url_for("main.landing"))


@bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        u = User.query.filter_by(
            email=form.email.data.strip().lower(), username=form.username.data.strip()
        ).first()
        if not u:
            flash("No account matches that email and username.", "danger")
            return render_template("auth/forgot.html", form=form)
        return redirect(url_for("auth.reset_password", uid=u.id))
    return render_template("auth/forgot.html", form=form)


@bp.route("/reset-password/<int:uid>", methods=["GET", "POST"])
def reset_password(uid):
    u = User.query.get_or_404(uid)
    form = ResetPasswordForm()
    if form.validate_on_submit():
        u.set_password(form.new_password.data)
        db.session.commit()
        flash("Password updated. Please sign in.", "success")
        return redirect(url_for("auth.login"))
    return render_template("auth/reset.html", form=form, user=u)
