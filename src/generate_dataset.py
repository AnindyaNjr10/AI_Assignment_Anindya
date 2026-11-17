"""
Creates the synthetic support-ticket dataset (data/tickets.csv).

The assignment allows us to create our own dataset. Descriptions were written
by hand (formal, casual, short, long, with typos) for 6 categories. A few
cross-category "tricky" tickets are included on purpose so the model is not
tested on perfectly separable data. A handful of duplicate rows, messy text and
missing values are injected so the cleaning step has real work to do.

Run:  python src/generate_dataset.py
"""
import random
from pathlib import Path
import pandas as pd

random.seed(42)

TICKETS = {
    "Login Issue": [
        "I forgot my password and cannot login",
        "Password reset email is not arriving",
        "Password reset link says it has expired",
        "Unable to sign in, it says invalid username or password",
        "My account is locked after too many failed attempts",
        "I did not receive the OTP for login",
        "The OTP code is wrong every time I enter it",
        "Cannot log into the portal from my phone",
        "Login page keeps refreshing without letting me in",
        "I am logged out automatically every few minutes",
        "Two factor authentication code is not working",
        "Forgot username, how do I recover it",
        "Sign in button does nothing when I click it",
        "Getting 'access denied' even with the correct password",
        "Reset password option is not working",
        "i cant login to my account plz help",
        "Single sign on fails and redirects me back to the login screen",
        "New password is rejected, says it was used before",
        "My login worked yesterday but today it says user not found",
        "Authentication failed while signing in to the dashboard",
        "Captcha on the login page never loads so I cannot sign in",
        "Account suspended message appears when I try to log in",
        "I keep getting a wrong credentials error although I typed it correctly",
        "Unable to access my account, password not accepted",
        "Verification code for sign in never reaches my email",
        "Locked out of the application since this morning",
        "How can I unlock my user account, I entered the wrong password thrice",
        "Cannot login using Google sign in option",
        "Session expired message shows up immediately after login",
        "Need help logging in, the password reset page gives a blank screen",
        "Login error: your account is temporarily disabled",
        "Not able to authenticate with my company email",
    ],
    "Application Error": [
        "The application is showing an error while saving data",
        "Getting error 500 when I submit the form",
        "App crashes every time I open the settings page",
        "Unexpected error occurred, please contact administrator",
        "Save button throws a server error",
        "The page shows a blank white screen after clicking submit",
        "I get a null pointer exception on the checkout screen",
        "Error message: failed to update record",
        "Application freezes and then closes by itself",
        "File upload fails with an unknown error",
        "Uploaded attachments are not saving, it shows error code 403",
        "The app stopped working after the latest update",
        "Data is not getting saved and a popup says something went wrong",
        "Getting a 404 page when I click on the dashboard link",
        "Submit form gives internal server error",
        "app keeps crashing on startup",
        "Search feature throws an exception and returns nothing",
        "Delete option gives an error and the entry stays there",
        "Error while exporting the file: operation could not be completed",
        "There is a bug, the total amount is calculated wrong on the invoice screen",
        "Dropdown list is empty and console shows a javascript error",
        "Application displays 'database connection failed' message",
        "Cannot save my profile, a red error banner appears",
        "The system throws an error whenever I try to attach a document",
        "Mobile app closes unexpectedly when I tap on notifications",
        "Form validation error appears even though all fields are filled",
        "Something went wrong message is displayed on every action",
        "The update button is not working and an error is thrown",
        "Fatal error on loading the user list page",
        "Unable to create new record, system returns an error",
        "Application error while processing payment page, transaction failed with exception",
        "The application is showing a strange error message while saving",
    ],
    "Report": [
        "Please help me generate my monthly sales report",
        "Need the quarterly revenue report for the finance team",
        "How do I download last month's attendance report",
        "I want a report of all closed tickets this week",
        "Can you add a new column to the weekly summary report",
        "The exported report is missing some rows",
        "Please share the annual performance report of the sales team",
        "Need a custom report showing orders by region",
        "Report shows wrong totals compared to the dashboard",
        "How to schedule the daily report to be emailed automatically",
        "Request for inventory stock report as of yesterday",
        "I need the customer list exported as an excel report",
        "Can I get a PDF version of the monthly report",
        "Looking for an expense summary report for the last quarter",
        "Need help to filter the report by date range",
        "send me the sales report for october",
        "Generate user activity report for the last 30 days",
        "The report template needs to include the company logo",
        "Requesting a data extract of all invoices raised in September",
        "Where can I find the profit and loss statement report",
        "Please create a dashboard report for management review",
        "Need a breakdown report of tickets by category",
        "Report format needs to be changed to landscape",
        "I would like to receive the weekly analytics report every Monday",
        "Need a report on employee leave balances",
        "Compare this quarter's report with the previous quarter",
        "Request to generate delivery status report for all branches",
        "The monthly report is not showing data for the north region",
        "Can you generate a report for customer complaints in the last six months",
        "Please provide a summary report of website visits",
        "Need sales figures in a report format for the board meeting",
        "Help me to create the yearly report",
    ],
    "Account Update": [
        "I need to change my registered mobile number",
        "Please update my email address on the account",
        "Want to change the name on my profile, there is a spelling mistake",
        "How do I update my billing address",
        "Need to change my account username",
        "Please add a second phone number to my account",
        "I moved to a new city, update my address in the system",
        "Change the contact person for our company account",
        "Update my date of birth, it was entered incorrectly",
        "I want to change my notification preferences",
        "Please deactivate my old email id and use the new one",
        "Need to change the account type from basic to premium",
        "Remove my previous phone number from the profile",
        "Please update the company name on our account",
        "Want to edit my profile picture and bio",
        "need to change my mobile no. registered with account",
        "Add another user to our team account",
        "I want to delete my account permanently",
        "Please transfer the account ownership to my colleague",
        "Change the language setting of my profile to Hindi",
        "Update GST number on our account details",
        "Kindly modify my registered email, the old one is not in use",
        "Request to change my shipping address",
        "Update my designation and department in the profile",
        "Please correct my surname in the account records",
        "I need to link my new bank account to my profile",
        "Change the primary contact number for my account",
        "How can I change the email id linked to my account",
        "I would like to update my personal details",
        "Please merge my two accounts into one",
        "Need my account details edited, my mobile number has changed",
        "Please update my profile information with the new address",
    ],
    "Performance": [
        "The application is very slow today",
        "Pages are taking too long to load",
        "The system is lagging badly during peak hours",
        "Dashboard takes more than a minute to open",
        "Search results are extremely slow",
        "App is running slowly since the last release",
        "High response time while saving records",
        "Everything is loading so slowly, I cannot work",
        "The website is slow and times out frequently",
        "Screen freezes for several seconds before responding",
        "It takes forever to open the customer page",
        "System performance has degraded in the last two days",
        "Mobile app is sluggish and takes ages to open",
        "Very long loading time when I click on any menu",
        "Application hangs for a long time on large files",
        "app is super slow plz fix",
        "Timeout occurs when many users are online at the same time",
        "The portal speed has become really bad this week",
        "Video and images load very slowly inside the application",
        "Page load time is more than 30 seconds",
        "Transactions are processing slowly and users are waiting",
        "System is very laggy while typing in the notes field",
        "Opening any record takes too much time",
        "The software is too slow for daily usage, kindly improve speed",
        "Uploading is taking a very long time even for small files",
        "Delays and slowness observed in the evening",
        "Our team is facing slow response from the application since morning",
        "Navigation between pages is slow",
        "The application takes too long to load after login",
        "Heavy lag and slow loading of the home screen",
        "Application not responding for long periods, then recovers",
        "Why is the application taking so much time to respond",
    ],
    "Billing Issue": [
        "I was charged twice for the same subscription",
        "Please send me the invoice for last month",
        "My refund has not been credited yet",
        "The amount on my bill is higher than expected",
        "Need a copy of the payment receipt",
        "I want to cancel my subscription and get a refund",
        "Wrong GST amount was charged on my invoice",
        "Payment was deducted but the order shows as unpaid",
        "How do I change my credit card for billing",
        "I was billed after I cancelled my plan",
        "Discount coupon was not applied to my invoice",
        "Please explain the extra charges on my statement",
        "Need an invoice with our company details for tax purposes",
        "Auto renewal charged me without any notice",
        "refund for double payment pls",
        "The payment failed but money has left my bank account",
        "Please correct the invoice, the plan name is wrong",
        "I upgraded my plan but the old price is still being charged",
        "Need to know the due date of my next payment",
        "Billing cycle date needs to be changed",
        "I have not received my invoice for this quarter",
        "Unrecognised charge on my card from your company",
        "Please apply my credit balance to the next bill",
        "Overcharged for additional users on the team plan",
        "Payment gateway shows success but subscription is not active",
        "Request a refund because the service was unavailable for two days",
        "How can I pay my pending dues by bank transfer",
        "The invoice total does not match the quotation",
        "Charged for premium plan though I am on the basic plan",
        "Kindly share the payment history for this year",
        "Please waive the late payment fee",
        "I need my previous invoices as a single document",
    ],
}

# Deliberately tricky tickets: vocabulary overlaps with another category.
TRICKY = [
    ("Login page is very slow to load", "Performance"),
    ("Report takes too long to generate", "Performance"),
    ("The report page shows an error when I click download", "Application Error"),
    ("I cannot login after changing my email address", "Login Issue"),
    ("Error while updating my mobile number in the profile", "Application Error"),
    ("Slow response when I open the billing page", "Performance"),
    ("Invoice report is not downloading, error appears", "Application Error"),
    ("Need to update the card details because the payment keeps failing", "Billing Issue"),
    ("Password changed but now the app gives an error at login", "Login Issue"),
    ("Monthly billing report required for the finance team", "Report"),
]

PRIORITY_BIAS = {
    "Login Issue": ["High", "High", "Medium"],
    "Application Error": ["High", "High", "Medium"],
    "Report": ["Low", "Medium", "Medium"],
    "Account Update": ["Low", "Low", "Medium"],
    "Performance": ["High", "Medium", "Medium"],
    "Billing Issue": ["Medium", "High", "Medium"],
}
STATUSES = ["Open", "Closed", "Closed", "In Progress", "Resolved"]


def build() -> pd.DataFrame:
    rows = [(d, c) for c, ds in TICKETS.items() for d in ds] + TRICKY
    random.shuffle(rows)
    data = []
    for i, (desc, cat) in enumerate(rows, start=1):
        data.append({
            "ticket_id": f"T{i:03d}",
            "ticket_description": desc,
            "category": cat,
            "priority": random.choice(PRIORITY_BIAS[cat]),
            "status": random.choice(STATUSES),
        })
    df = pd.DataFrame(data)

    # --- Inject realistic data-quality problems -----------------------
    # 1) a few exact duplicate tickets (customers re-submitting the same request)
    dup = df.sample(6, random_state=1).copy()
    dup["ticket_id"] = [f"T{len(df) + k + 1:03d}" for k in range(len(dup))]
    df = pd.concat([df, dup], ignore_index=True)
    # 2) messy casing / whitespace / punctuation on a few rows
    for idx in df.sample(12, random_state=2).index:
        df.loc[idx, "ticket_description"] = "  " + str(df.loc[idx, "ticket_description"]).upper() + "!!  "
    # 3) a few missing values
    df.loc[df.sample(3, random_state=3).index, "priority"] = None
    df.loc[df.sample(3, random_state=4).index, "status"] = None
    df.loc[df.sample(2, random_state=5).index, "ticket_description"] = None
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "tickets.csv"
    df = build()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(df["category"].value_counts())
