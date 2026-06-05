RAG_DOCUMENTS = [
    # 1. Refund Policy
    """Acme Corp Refund Policy.
    All standard plan customers are eligible for a full refund within 30 days of purchase.
    Enterprise plan customers receive an extended 60-day refund window with pro-rated refunds
    calculated from the date of cancellation. Refunds are processed within 5-7 business days
    and returned to the original payment method. No refunds are available after the refund
    window closes. Customers must submit refund requests through the support portal or by
    contacting their account manager directly. Annual subscriptions that are cancelled mid-term
    will receive a pro-rated credit for the remaining months.""",

    # 2. Product Tiers
    """Acme Corp Product Overview.
    Acme Corp offers three product tiers: Starter, Professional, and Enterprise.
    The Starter plan includes basic features for individual users at $29 per month.
    The Professional plan adds team collaboration, advanced analytics, and priority
    support for $99 per month per user. The Enterprise plan includes everything in
    Professional plus custom integrations, dedicated account management, SSO,
    audit logs, and a 99.99% uptime SLA. Enterprise pricing is custom and starts
    at $500 per month for up to 50 users. All plans include a 14-day free trial
    with no credit card required.""",

    # 3. Security
    """Acme Corp Security Practices.
    Acme Corp maintains SOC 2 Type II compliance and undergoes annual third-party
    security audits. All data is encrypted at rest using AES-256 and in transit
    using TLS 1.3. Customer data is stored in isolated tenants within AWS
    us-east-1 and eu-west-1 regions. Data residency can be configured per
    organization for Enterprise customers. Backups are performed every 6 hours
    with 30-day retention. Acme Corp does not sell or share customer data with
    third parties. Enterprise customers can request data deletion within 24 hours.
    Bug bounty program available through HackerOne.""",

    # 4. API Docs
    """Acme Corp API Documentation.
    The Acme API uses REST with JSON request and response bodies. Authentication
    is via Bearer tokens issued through OAuth 2.0. Rate limits are 100 requests
    per minute for Starter, 1000 for Professional, and 10000 for Enterprise.
    Rate limit headers are included in every response: X-RateLimit-Limit,
    X-RateLimit-Remaining, and X-RateLimit-Reset. Exceeding the rate limit
    returns HTTP 429 with a Retry-After header. The API supports pagination
    via cursor-based pagination using the next_cursor field. Webhooks are
    available for real-time event notifications on Professional and Enterprise
    plans. API versioning uses date-based versions in the URL path.""",

    # 5. Uptime SLA
    """Acme Corp Uptime and Reliability.
    Acme Corp guarantees 99.9% uptime for Professional plans and 99.99% uptime
    for Enterprise plans. Uptime is calculated monthly excluding scheduled
    maintenance windows which are announced 72 hours in advance. If uptime
    falls below the guaranteed level, customers receive service credits:
    10% credit for each 0.1% below the SLA threshold, up to a maximum of
    30% of the monthly fee. Service credits must be requested within 30 days
    of the incident. Status page updates are posted at status.acme.com
    within 5 minutes of any detected incident. Post-incident reports are
    published within 48 hours for any outage exceeding 15 minutes.""",

    # 6. Team Directory
    """Acme Corp Leadership and Key Contacts.
    Jane Doe serves as the Chief Executive Officer (CEO) of Acme Corp. John Smith is
    the Chief Technology Officer (CTO), overseeing all engineering and product development.
    Sarah Jenkins is the VP of Product, responsible for roadmap planning and customer feedback.
    Mike Davis leads the Customer Support department as Director of Support. Legal inquiries
    should be directed to legal@acme.com, which is managed by General Counsel Alice Vance.""",

    # 7. Customer Support Hours
    """Acme Corp Support Hours and Response Times.
    Standard customer support is available Monday through Friday from 9:00 AM to 5:00 PM EST.
    Enterprise customers receive 24/7/365 priority support for critical (P0/P1) incidents.
    The target response time for P0 (critical outage) incidents is under 15 minutes.
    For Professional plan users, support is available 12 hours a day, 5 days a week, with a
    target response time of under 4 hours for any support ticket.""",

    # 8. Billing and Invoicing
    """Acme Corp Billing and Invoice Terms.
    Starter and Professional plans are billed via credit card (Visa, Mastercard, AMEX) or PayPal.
    Enterprise customers can be invoiced monthly or annually with payment terms of Net 30.
    Annual subscriptions paid upfront receive a 20% discount compared to monthly billing.
    Late payments are subject to a 1.5% monthly fee on outstanding balances after a 5-day
    grace period. All invoicing questions should be sent to billing@acme.com.""",

    # 9. Technical Support Channels
    """Acme Corp Technical Support Channels.
    Customers on the Starter plan can access support solely through email at support@acme.com.
    Professional plan customers have access to email support plus live chat within the dashboard.
    Enterprise plan customers receive a dedicated Slack channel with shared channels (Slack Connect)
    and can schedule emergency phone calls with a designated Solutions Engineer or account manager.""",

    # 10. Data Retention
    """Acme Corp Data Retention and Deletion.
    Application logs are retained for 30 days for debugging purposes and then permanently deleted.
    Active customer database data is retained indefinitely while the account remains active.
    Upon account cancellation, customer databases and configurations are soft-deleted for 14 days,
    after which they are hard-deleted from AWS servers. Offsite database backups are kept for 90 days.""",

    # 11. Password Policy
    """Acme Corp Password Security Requirements.
    All user accounts must have passwords containing a minimum of 12 characters.
    Passwords must include at least one uppercase letter, one lowercase letter, one number,
    and one special character (e.g., !, @, #, $, %, ^, &, *). Password changes are enforced
    every 90 days. Multi-Factor Authentication (MFA) is optional but recommended for Starter
    and Professional plans, and mandatory for all Enterprise users.""",

    # 12. GDPR and Privacy
    """Acme Corp GDPR and Privacy Compliance.
    Acme Corp is fully GDPR and CCPA compliant. Customers can request an export of their personal data
    by emailing privacy@acme.com or using the self-service export tool in settings. Data deletion
    requests can be submitted similarly. Our Data Protection Officer (DPO) is Robert Lin, reachable
    at dpo@acme.com. We do not track users across external sites, and cookie settings can be adjusted.""",

    # 13. Office Locations
    """Acme Corp Corporate Office Locations.
    The global headquarters of Acme Corp is located at 100 Pine Street, San Francisco, CA 94111, USA.
    Our European regional office is based at 42 Canary Wharf, London E14 5AB, United Kingdom.
    Our Asia-Pacific operations are managed from our regional office at 1-chome, Minato-ku,
    Tokyo 105-0011, Japan. We operate with a hybrid-first employee distribution model.""",

    # 14. Observed Holidays
    """Acme Corp Observed Corporate Holidays.
    Our offices are closed on the following US holidays: New Year's Day, Martin Luther King Jr. Day,
    Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving Day, the Day after Thanksgiving,
    Christmas Eve, and Christmas Day. Support response times during observed holidays may be slightly
    extended for Starter and Professional plan customers, but Enterprise support remains active.""",

    # 15. Employee Benefits
    """Acme Corp Employee Benefits Overview.
    Acme Corp offers 100% employer-covered health, dental, and vision insurance for full-time employees
    and their dependents. We provide a 401(k) retirement plan with a 4% matching contribution.
    Employees receive unlimited Paid Time Off (PTO), an annual learning and development budget of $2000,
    and a $500 home office setup reimbursement to support remote workspace needs.""",

    # 16. Remote Work Policy
    """Acme Corp Remote Work and Core Hours.
    Employees are permitted to work remotely or from any of our offices. Regardless of location, all team
    members are expected to be online and available during core operating hours, which are defined as
    10:00 AM to 4:00 PM EST. Weekly company-wide updates (All Hands) are held on Wednesdays at 1:00 PM EST
    and attendance is mandatory for all salaried staff.""",

    # 17. System Maintenance
    """Acme Corp Scheduled System Maintenance.
    Routine system maintenance is scheduled every Tuesday and Thursday between 2:00 AM and 4:00 AM UTC.
    During these windows, minor service interruptions (under 2 minutes) may occur. Large-scale database
    migrations or software deployments are performed on Sundays between 12:00 AM and 4:00 AM UTC.
    All maintenance schedules are updated live at status.acme.com.""",

    # 18. Training and Onboarding
    """Acme Corp Onboarding and Education.
    New hires participate in a 2-week structured onboarding cohort covering company culture, engineering
    standards, and security policies. We offer self-paced learning paths through Acme Academy.
    Weekly engineering tech talks are held on Fridays at 12:00 PM EST, and recordings are archived
    on the internal wiki for remote developers in other timezones.""",

    # 19. Environmental Policy
    """Acme Corp Sustainability and Carbon Goals.
    Acme Corp is committed to achieving net-zero carbon emissions by the year 2028.
    All our server workloads are hosted on AWS green data centers that utilize 100% renewable energy.
    We implement corporate e-waste recycling and prohibit single-use plastics in all office locations.
    Employees are reimbursed up to $50 monthly for using public transit or bicycling to work.""",

    # 20. Corporate History
    """Acme Corp Corporate History and Funding.
    Acme Corp was founded in 2018 by Jane Doe and John Smith in San Francisco.
    The company raised a $3M Seed round in 2019, followed by a $12M Series A funding round in 2020
    led by Apex Ventures. Acme expanded operations globally by opening its London office in 2022
    and its Tokyo office in 2023. As of 2026, Acme serves over 10,000 corporate customers globally."""
]
