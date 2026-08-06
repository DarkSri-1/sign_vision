from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, PasswordField, SubmitField, BooleanField, TextAreaField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional, ValidationError


class SignupForm(FlaskForm):
    full_name = StringField("Full name", validators=[DataRequired(), Length(max=120)])
    email = StringField("Email", validators=[DataRequired(), Email(), Length(max=120)])
    username = StringField("Username", validators=[DataRequired(), Length(min=3, max=64)])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=6, max=128)])
    confirm_password = PasswordField(
        "Confirm password",
        validators=[DataRequired(), EqualTo("password", message="Passwords must match.")],
    )
    submit = SubmitField("Create account")


class LoginForm(FlaskForm):
    username_or_email = StringField("Username or email", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    remember = BooleanField("Remember me")
    submit = SubmitField("Sign in")


class ForgotPasswordForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    username = StringField("Username", validators=[DataRequired()])
    submit = SubmitField("Verify identity")


class ResetPasswordForm(FlaskForm):
    new_password = PasswordField("New password", validators=[DataRequired(), Length(min=6, max=128)])
    confirm = PasswordField(
        "Confirm new password",
        validators=[DataRequired(), EqualTo("new_password", message="Passwords must match.")],
    )
    submit = SubmitField("Update password")


class ProfileForm(FlaskForm):
    full_name = StringField("Full name", validators=[DataRequired(), Length(max=120)])
    username = StringField("Username", validators=[DataRequired(), Length(min=3, max=64)])
    current_password = PasswordField("Current password (required to change password)", validators=[Optional()])
    new_password = PasswordField("New password", validators=[Optional(), Length(min=6, max=128)])
    confirm_password = PasswordField(
        "Confirm new password",
        validators=[Optional(), EqualTo("new_password", message="Passwords must match.")],
    )
    submit = SubmitField("Save changes")


class HistorySearchForm(FlaskForm):
    class Meta:
        csrf = False

    date_from = StringField("From date", validators=[Optional()])
    date_to = StringField("To date", validators=[Optional()])
    sign = StringField("Predicted sign", validators=[Optional()])
    source = SelectField(
        "Source",
        choices=[
            ("", "All sources"),
            ("webcam", "Webcam"),
            ("upload", "File upload"),
        ],
        validators=[Optional()],
    )
    submit = SubmitField("Apply filters")


class RecognitionNoteForm(FlaskForm):
    user_note = TextAreaField(
        "Personal note",
        validators=[Optional(), Length(max=500)],
        render_kw={
            "rows": 4,
            "class": "form-control user-form-control",
            "placeholder": "Optional note for this recognition (practice goal, context, etc.)",
        },
    )
    submit = SubmitField("Save note")


class DatasetUploadForm(FlaskForm):
    class_name = StringField("Class / sign name", validators=[DataRequired(), Length(max=64)])
    images = FileField(
        "Images",
        validators=[
            DataRequired(),
            FileAllowed(["jpg", "jpeg", "png", "webp", "bmp"], "Images only."),
        ],
    )
    submit = SubmitField("Upload")


class TrainForm(FlaskForm):
    epochs = StringField("Epochs", validators=[Optional()])
    submit = SubmitField("Start training")


class AdminUserToggleForm(FlaskForm):
    submit = SubmitField("Update")


class AdminUserCreateForm(FlaskForm):
    full_name = StringField("Full name", validators=[DataRequired(), Length(max=120)])
    email = StringField("Email", validators=[DataRequired(), Email(), Length(max=120)])
    username = StringField("Username", validators=[DataRequired(), Length(min=3, max=64)])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=6, max=128)])
    confirm_password = PasswordField(
        "Confirm password",
        validators=[DataRequired(), EqualTo("password", message="Passwords must match.")],
    )
    is_admin = BooleanField("Grant administrator role", default=False)
    submit = SubmitField("Create user")


class AdminUserEditForm(FlaskForm):
    full_name = StringField("Full name", validators=[DataRequired(), Length(max=120)])
    email = StringField("Email", validators=[DataRequired(), Email(), Length(max=120)])
    username = StringField("Username", validators=[DataRequired(), Length(min=3, max=64)])
    is_admin = BooleanField("Administrator", default=False)
    is_active = BooleanField("Account active", default=True)
    new_password = PasswordField("New password (optional)", validators=[Optional(), Length(min=6, max=128)])
    confirm_password = PasswordField(
        "Confirm new password",
        validators=[Optional(), EqualTo("new_password", message="Passwords must match.")],
    )
    submit = SubmitField("Save changes")


class AdminHistoryFilterForm(FlaskForm):
    class Meta:
        csrf = False

    user_id = StringField("User ID", validators=[Optional()])
    date_from = StringField("From date", validators=[Optional()])
    date_to = StringField("To date", validators=[Optional()])
    sign = StringField("Predicted sign", validators=[Optional()])
    source = SelectField(
        "Source",
        choices=[
            ("", "All sources"),
            ("webcam", "Webcam"),
            ("upload", "File upload"),
        ],
        validators=[Optional()],
    )
    submit = SubmitField("Apply filters")


class AdminLogFilterForm(FlaskForm):
    class Meta:
        csrf = False

    user_id = StringField("User ID", validators=[Optional()])
    submit = SubmitField("Filter")


class AdminDatasetRenameForm(FlaskForm):
    old_name = StringField("Current class name", validators=[DataRequired(), Length(max=120)])
    new_name = StringField("New class name", validators=[DataRequired(), Length(max=120)])
    submit = SubmitField("Rename folder")


class AdminRecognitionNoteForm(FlaskForm):
    user_note = TextAreaField(
        "Note (visible to user on their history)",
        validators=[Optional(), Length(max=500)],
        render_kw={
            "rows": 3,
            "class": "form-control",
            "placeholder": "Optional admin/user-visible note…",
        },
    )
    submit = SubmitField("Save note")
