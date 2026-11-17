"""Bonus task: rule-based suggested replies for each predicted category."""

RESPONSES = {
    "Login Issue": (
        'Please use the "Forgot Password" option on the login page to reset your password. '
        "If your account is locked, wait 15 minutes and try again. If the problem continues, "
        "please contact the support team."
    ),
    "Application Error": (
        "Sorry about the trouble. Please refresh the page and try again. If the error persists, "
        "reply with a screenshot and the exact error message so our technical team can investigate."
    ),
    "Report": (
        "You can generate standard reports from Reports > Create Report. For a custom report, "
        "please share the date range and fields you need and we will prepare it for you."
    ),
    "Account Update": (
        "You can edit your details under Profile > Account Settings. For security reasons, "
        "changes to your mobile number or email may need OTP verification."
    ),
    "Performance": (
        "We are sorry for the slowness. Please try clearing your browser cache or switching browsers. "
        "Our team has been notified and is checking system performance."
    ),
    "Billing Issue": (
        "Thank you for reaching out. Our billing team will review your invoice and payment details "
        "and get back to you within 1-2 business days. Refunds, where applicable, take 5-7 working days."
    ),
}


def suggested_response(category: str) -> str:
    return RESPONSES.get(category, "Thank you for contacting us. A support agent will get back to you shortly.")
