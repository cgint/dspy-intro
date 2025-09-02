from typing import Dict, List, Set
import dspy
import random

# ---- start train-set from codegiant -----

examples_for_classification_train_set_giant: List[Dict[str, str]] = [
    # Unsafe: Plaintext User/Password
    {
        "classify_input": "Hello, here is my user 'cgint' and password: 12356!",
        "classification": "unsafe"
    },
    {
        "classify_input": "My login is user@example.com and my password is Password123!",
        "classification": "unsafe"
    },
    {
        "classify_input": "Here are the credentials: username=testuser, password=testpassword",
        "classification": "unsafe"
    },
    {
        "classify_input": "For the FTP server, use ftp_user and P@ssw0rd!",
        "classification": "unsafe"
    },
    {
        "classify_input": "The account details are 'john.doe' with the temporary password 'changeme'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "ssh login: root@192.168.1.1, password is toor",
        "classification": "unsafe"
    },
    {
        "classify_input": "Just use my credentials for now: alice / 12345678",
        "classification": "unsafe"
    },
    {
        "classify_input": "The default password for the router is 'admin123'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "Here's the login for the staging server: stage_user / St@g!ngP@ss",
        "classification": "unsafe"
    },
    {
        "classify_input": "The wifi password is 'GuestNetwork123'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "For the database, the user is 'db_admin' and the password is 'dbs3cr3t'.",
        "classification": "unsafe"
    },
    # Unsafe: API Keys and Tokens
    {
        "classify_input": "My OpenAI API key is sk-aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890.",
        "classification": "unsafe"
    },
    {
        "classify_input": "Use this bearer token: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.ZRrHA19WJ2ACtGgYgdHHd3xG8I5iPYZ_trc_yI-tX2A",
        "classification": "unsafe"
    },
    {
        "classify_input": "The GitHub token is ghp_xYzAbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGh.",
        "classification": "unsafe"
    },
    {
        "classify_input": "Here's the Slack bot token: xoxb-1234567890-abcdefghijklmnopqrstuvwxyz.",
        "classification": "unsafe"
    },
    {
        "classify_input": "Set the STRIPE_API_KEY to pk_live_xxxxxxxxxxxxxxxxxxxxxxxx.",
        "classification": "unsafe"
    },
    {
        "classify_input": "The authorization key is 'Authorization: Basic YWxhZGRpbjpvcGVuc2VzYW1l'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "My AWS access key ID is AKIAIOSFODNN7EXAMPLE.",
        "classification": "unsafe"
    },
    {
        "classify_input": "And the AWS secret access key is wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY.",
        "classification": "unsafe"
    },
    {
        "classify_input": "The API secret for the service is 1a2b3c4d-5e6f-7g8h-9i0j-k1l2m3n4o5p6.",
        "classification": "unsafe"
    },
    {
        "classify_input": "The client secret is 'my_super_secret_client_secret_123'.",
        "classification": "unsafe"
    },
    # Unsafe: Connection Strings
    {
        "classify_input": "DB connection string: postgresql://user:password@host:5432/mydatabase",
        "classification": "unsafe"
    },
    {
        "classify_input": "The connection URL is mysql://admin:P@ssw0rd_123@db.example.com/prod_db",
        "classification": "unsafe"
    },
    {
        "classify_input": "The MongoDB connection string is mongodb+srv://myuser:myPassword@cluster0.mongodb.net/test",
        "classification": "unsafe"
    },
    # Unsafe: Private Keys
    {
        "classify_input": "-----BEGIN RSA PRIVATE KEY-----\nMIIE... (rest of the key) ...-----END RSA PRIVATE KEY-----",
        "classification": "unsafe"
    },
    {
        "classify_input": "-----BEGIN PGP PRIVATE KEY BLOCK-----\n...key data...\n-----END PGP PRIVATE KEY BLOCK-----",
        "classification": "unsafe"
    },
    # Unsafe: PII (Personally Identifiable Information)
    {
        "classify_input": "My credit card number is 4111-1111-1111-1111.",
        "classification": "unsafe"
    },
    {
        "classify_input": "My SSN is 123-45-6789.",
        "classification": "unsafe"
    },
    {
        "classify_input": "Please use my Amex: 3782 822463 10005.",
        "classification": "unsafe"
    },
    {
        "classify_input": "My full card details are Visa 4916123456789012, expiry 12/25, CVV 123.",
        "classification": "unsafe"
    },
    # Unsafe: Hardcoded secrets in code
    {
        "classify_input": "config['password'] = 's3cr3t_p4ss'",
        "classification": "unsafe"
    },
    {
        "classify_input": "const API_KEY = \"a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6\";",
        "classification": "unsafe"
    },
    {
        "classify_input": "define('AUTH_SALT', '`+>b~U+c2N%gP_b|p(y^+JGBR(q@f`f_a_`H@Vb`j~+g`_e`H@Vb`j~+g');",
        "classification": "unsafe"
    },
    {
        "classify_input": "let secret = 'my-jwt-secret';",
        "classification": "unsafe"
    },
    {
        "classify_input": "private static final String SECRET_KEY = \"my_legacy_secret_key\";",
        "classification": "unsafe"
    },
    # Unsafe: More variations
    {
        "classify_input": "user: admin, pass: root",
        "classification": "unsafe"
    },
    {
        "classify_input": "The credentials for the test account are test_user:p4ssw0rd.",
        "classification": "unsafe"
    },
    {
        "classify_input": "Connect with this API key: 9876543210fedcba9876543210fedcba.",
        "classification": "unsafe"
    },
    {
        "classify_input": "Here's a temporary access token: temp_token_abcdef123456.",
        "classification": "unsafe"
    },
    {
        "classify_input": "For the Google Maps API, use key AIzaSyA... (full key here).",
        "classification": "unsafe"
    },
    {
        "classify_input": "The webhook secret is whsec_abcdef1234567890.",
        "classification": "unsafe"
    },
    {
        "classify_input": "My Twitter API secret is TwiTtErSeCrEtKeY12345.",
        "classification": "unsafe"
    },
    {
        "classify_input": "Here's the JWT for auth: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "classification": "unsafe"
    },
    {
        "classify_input": "I set up a new account. It's 'newbie' and the pass is 'newbiepass'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "My email is bob@work.com and the password is mydog'snameFido",
        "classification": "unsafe"
    },
    # Safe: General Conversation & Benign Keywords
    {
        "classify_input": "Hello, how are you today?",
        "classification": "safe"
    },
    {
        "classify_input": "Can you help me debug this Python script? It's a simple bubble sort algorithm.",
        "classification": "safe"
    },
    {
        "classify_input": "What's the best way to handle file uploads in Flask?",
        "classification": "safe"
    },
    {
        "classify_input": "The key to a good API is consistent design.",
        "classification": "safe"
    },
    {
        "classify_input": "I'm getting a 'permission denied' error, but I'm not sure why.",
        "classification": "safe"
    },
    {
        "classify_input": "The user interface needs to be more intuitive.",
        "classification": "safe"
    },
    {
        "classify_input": "This code block calculates the factorial of a number.",
        "classification": "safe"
    },
    {
        "classify_input": "I need to write a function that takes a username and validates it against a regex.",
        "classification": "safe"
    },
    {
        "classify_input": "The secret to making good bread is patience.",
        "classification": "safe"
    },
    {
        "classify_input": "I need a token to play this arcade game.",
        "classification": "safe"
    },
    {
        "classify_input": "The main key in this dictionary is 'data'.",
        "classification": "safe"
    },
    {
        "classify_input": "This document requires a password to open.",
        "classification": "safe"
    },
    {
        "classify_input": "The user needs to be authenticated.",
        "classification": "safe"
    },
    {
        "classify_input": "This is a base64 encoded string of 'hello world': aGVsbG8gd29ybGQ=",
        "classification": "safe"
    },
    {
        "classify_input": "My user ID is 12345.",
        "classification": "safe"
    },
    {
        "classify_input": "The key signature for this piece of music is C Major.",
        "classification": "safe"
    },
    {
        "classify_input": "The password for the puzzle is 'knowledge'.",
        "classification": "safe"
    },
    {
        "classify_input": "The function is called `get_user_password_hash()`.",
        "classification": "safe"
    },
    {
        "classify_input": "The database schema has a 'password_hash' column.",
        "classification": "safe"
    },
    {
        "classify_input": "The security token grants access to the event.",
        "classification": "safe"
    },
    {
        "classify_input": "The user's name is Robert'); DROP TABLE Students;--",
        "classification": "safe"
    },
    {
        "classify_input": "My username is 'test-user', can you check my account status?",
        "classification": "safe"
    },
    {
        "classify_input": "The API key format is usually a long alphanumeric string.",
        "classification": "safe"
    },
    {
        "classify_input": "You need to provide a username and password to log in.",
        "classification": "safe"
    },
    {
        "classify_input": "The variable `password` should be handled with care.",
        "classification": "safe"
    },
    # Safe: Redacted or Abstracted Secrets
    {
        "classify_input": "My password is '**********'.",
        "classification": "safe"
    },
    {
        "classify_input": "The API key is sk-....-xxxx.",
        "classification": "safe"
    },
    {
        "classify_input": "Please use the credentials: user='testuser', password='[REDACTED]'.",
        "classification": "safe"
    },
    {
        "classify_input": "The DB connection string is postgresql://user:[PASSWORD]@host:5432/db.",
        "classification": "safe"
    },
    {
        "classify_input": "My credit card ends in 1234.",
        "classification": "safe"
    },
    {
        "classify_input": "The token is `ghp_...`.",
        "classification": "safe"
    },
    {
        "classify_input": "I've reset your password. It is now `--omitted--`.",
        "classification": "safe"
    },
    {
        "classify_input": "The secret is stored in an environment variable, not here.",
        "classification": "safe"
    },
    {
        "classify_input": "My username is 'admin', but I can't share the password.",
        "classification": "safe"
    },
    {
        "classify_input": "The AWS secret key is set to '***'.",
        "classification": "safe"
    },
    {
        "classify_input": "My SSN is XXX-XX-1234.",
        "classification": "safe"
    },
    {
        "classify_input": "The client secret is `<SECRET_HIDDEN>`.",
        "classification": "safe"
    },
    {
        "classify_input": "The password for the wifi is written on the board.",
        "classification": "safe"
    },
    {
        "classify_input": "I've put the API key in the `.env` file for security.",
        "classification": "safe"
    },
    {
        "classify_input": "The user is 'test' and the password is 'test'.",
        "classification": "safe"
    },
    {
        "classify_input": "The default password is 'password', you should change it immediately.",
        "classification": "safe"
    },
    {
        "classify_input": "The key is `pk_test_...` which is a test key and not sensitive.",
        "classification": "safe"
    },
    {
        "classify_input": "The user is 'john.doe' and his password has been sent to him via a secure channel.",
        "classification": "safe"
    },
    {
        "classify_input": "The API key is loaded from `os.environ.get('API_KEY')`.",
        "classification": "safe"
    },
    {
        "classify_input": "This is a private key, but it's an example from a textbook: `-----BEGIN EXAMPLE PRIVATE KEY-----`",
        "classification": "safe"
    },
    {
        "classify_input": "The user's password must be at least 8 characters long.",
        "classification": "safe"
    },
    {
        "classify_input": "The API requires a key for authentication.",
        "classification": "safe"
    },
    {
        "classify_input": "My user account is locked.",
        "classification": "safe"
    },
    {
        "classify_input": "The classification of this text is 'safe'.",
        "classification": "safe"
    },
    {
        "classify_input": "The secret code to enter the clubhouse is 'friend'.",
        "classification": "safe"
    },
]

# ---- end train-set from codegiant -----

# ---- start test-set from codegiant -----

examples_for_classification_test_set_giant: List[Dict[str, str]] = [
    # Unsafe: User/Password variations
    {
        "classify_input": "login: myuser / mypass123",
        "classification": "unsafe"
    },
    {
        "classify_input": "The credentials are: user='app_user', pass='S3cureP@ssw0rd!'",
        "classification": "unsafe"
    },
    {
        "classify_input": "For the test environment, the user is 'tester' and the password is 'password'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "My wifi password is 'LiveLaughLove2024'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "The root password for the server is 'r00t_p@ssw0rd'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "Use this for the demo: demo_user / Demo12345",
        "classification": "unsafe"
    },
    {
        "classify_input": "The password for the zip file is 'secret'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "My email is jane.d@email.com and my password is hercat'snameWhiskers",
        "classification": "unsafe"
    },
    {
        "classify_input": "The admin account is 'administrator' with password 'adminpass'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "Here's the login for the production database: prod_user / Pr0d!P@ssw0rd",
        "classification": "unsafe"
    },
    # Unsafe: API Keys/Tokens variations
    {
        "classify_input": "The SendGrid API Key is SG.xxxxxxxx.yyyyyyyyyyyyyyyy.",
        "classification": "unsafe"
    },
    {
        "classify_input": "Here's the auth token: 'Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b'",
        "classification": "unsafe"
    },
    {
        "classify_input": "My Twilio Account SID is ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx and Auth Token is yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy.",
        "classification": "unsafe"
    },
    {
        "classify_input": "The private key for the service account is: '-----BEGIN PRIVATE KEY-----\n...key data...\n-----END PRIVATE KEY-----'",
        "classification": "unsafe"
    },
    {
        "classify_input": "The Mapbox access token is pk.eyJ1IjoibXl1c2VybmFtZSI.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.",
        "classification": "unsafe"
    },
    {
        "classify_input": "The client_secret for the OAuth flow is 'a1b2c3d4e5f6g7h8i9j0'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "The webhook signing secret is `whsec_aBcDeFgHiJkLmNoPqRsTuVwXyZ`",
        "classification": "unsafe"
    },
    {
        "classify_input": "My AWS_ACCESS_KEY_ID is AKIADEADBEEF... and my AWS_SECRET_ACCESS_KEY is 1234567890...",
        "classification": "unsafe"
    },
    {
        "classify_input": "The bearer token for the API is: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJPbmxpbmUgSldUIEJ1aWxkZXIiLCJpYXQiOjE2MTYyMzkwMjIsImV4cCI6MTY0NzcyNjYyMiwiYXVkIjoid3d3LmV4YW1wbGUuY29tIiwic3ViIjoianJvY2tldEBleGFtcGxlLmNvbSIsIkdpdmVuTmFtZSI6IkpvaG5ueSIsIlN1cm5hbWUiOiJSb2NrZXQiLCJFbWFpbCI6Impyb2NrZXRAZXhhbXBsZS5jb20iLCJSb2xlIjpbIk1hbmFnZXIiLCJQcm9qZWN0IEFkbWluaXN0cmF0b3IiXX0.7_I-3tC-g_t_I-3tC-g_t_I-3tC-g_t_I-3tC-g_t_I",
        "classification": "unsafe"
    },
    {
        "classify_input": "The secret key for the application is 'django-insecure-!@#$%^&*()'.",
        "classification": "unsafe"
    },
    # Unsafe: Connection Strings/PII/Keys variations
    {
        "classify_input": "My card is a Mastercard: 5412 7512 3412 3456, expires 06/24, CVV 456.",
        "classification": "unsafe"
    },
    {
        "classify_input": "The database connection is: 'mongodb://user:pass@host:port/db'",
        "classification": "unsafe"
    },
    {
        "classify_input": "My SSN is 987-65-4321, please don't share it.",
        "classification": "unsafe"
    },
    {
        "classify_input": "Here is the Redis connection URL: redis://:p@ssw0rd@redis-12345.c1.us-east-1-2.ec2.cloud.redislabs.com:12345",
        "classification": "unsafe"
    },
    {
        "classify_input": "-----BEGIN OPENSSH PRIVATE KEY-----\n...key data...\n-----END OPENSSH PRIVATE KEY-----",
        "classification": "unsafe"
    },
    {
        "classify_input": "My Visa number is 4111222233334444.",
        "classification": "unsafe"
    },
    {
        "classify_input": "The connection string for SQL Server is 'Server=myServerAddress;Database=myDataBase;User Id=myUsername;Password=myPassword;'",
        "classification": "unsafe"
    },
    {
        "classify_input": "My Amex card number is 3412 345678 91011.",
        "classification": "unsafe"
    },
    {
        "classify_input": "The authorization header is 'Basic dXNlcjpwYXNz'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "The credentials for the API are client_id: 'my_client_id', client_secret: 'my_client_secret'.",
        "classification": "unsafe"
    },
    # Unsafe: More variations
    {
        "classify_input": "I'm locked out. My user is 'm_smith' and my password is 'Summer2023!'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "The user is 'service_account' and the password is 'SvcAccP@ss1'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "My username is 'julie7' and my password is 'julie123'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "The default login for the device is user: 'support', pass: 'support01'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "Here is the API key for the weather service: 11223344556677889900aabbccddeeff.",
        "classification": "unsafe"
    },
    {
        "classify_input": "The secret for the JWT is 'this-is-a-very-secret-key-that-should-not-be-in-code'.",
        "classification": "unsafe"
    },
    {
        "classify_input": "My GitHub token is ghp_abc123def456... (full token).",
        "classification": "unsafe"
    },
    {
        "classify_input": "The API key is defined in the code: const API_KEY = 'my-secret-api-key';",
        "classification": "unsafe"
    },
    {
        "classify_input": "The Slack token is xoxp-12345-67890-abcde-fghij.",
        "classification": "unsafe"
    },
    {
        "classify_input": "The JWT is eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InRlc3QiLCJpYXQiOjE2MTYyMzkwMjJ9.4-Sz_I3o-Qo_I3o-Qo_I3o-Qo_I3o-Qo_I3o-Qo",
        "classification": "unsafe"
    },
    # Safe: General Conversation & Benign Keywords variations
    {
        "classify_input": "I think I found a bug in the password reset flow.",
        "classification": "safe"
    },
    {
        "classify_input": "This is a public API key, so it's safe to share: 'public_key_123'.",
        "classification": "safe"
    },
    {
        "classify_input": "How do I hash a password securely in PHP?",
        "classification": "safe"
    },
    {
        "classify_input": "The key insight here is that the algorithm is O(n log n).",
        "classification": "safe"
    },
    {
        "classify_input": "Let's talk about the new security policy.",
        "classification": "safe"
    },
    {
        "classify_input": "The user 'anonymous' has read-only access.",
        "classification": "safe"
    },
    {
        "classify_input": "The classification of this document is 'confidential'.",
        "classification": "safe"
    },
    {
        "classify_input": "I need a key to open this chest in the game.",
        "classification": "safe"
    },
    {
        "classify_input": "The secret to their success is a great team.",
        "classification": "safe"
    },
    {
        "classify_input": "The function `create_user` takes username and password as arguments.",
        "classification": "safe"
    },
    {
        "classify_input": "The database table 'users' has columns: 'id', 'username', 'email', 'password_hash'.",
        "classification": "safe"
    },
    {
        "classify_input": "The access control list (ACL) is misconfigured.",
        "classification": "safe"
    },
    {
        "classify_input": "The user's session token has expired.",
        "classification": "safe"
    },
    {
        "classify_input": "The primary key for the 'users' table is 'id'.",
        "classification": "safe"
    },
    {
        "classify_input": "This is a private repository.",
        "classification": "safe"
    },
    {
        "classify_input": "The secret ingredient in the sauce is a pinch of cinnamon.",
        "classification": "safe"
    },
    {
        "classify_input": "The user's password must meet the complexity requirements.",
        "classification": "safe"
    },
    {
        "classify_input": "The word 'password' appears 5 times in this document.",
        "classification": "safe"
    },
    {
        "classify_input": "My user ID is 54321.",
        "classification": "safe"
    },
    {
        "classify_input": "The key of the song is A minor.",
        "classification": "safe"
    },
    {
        "classify_input": "The password to the speakeasy is 'swordfish'.",
        "classification": "safe"
    },
    {
        "classify_input": "The function is named `check_password`.",
        "classification": "safe"
    },
    {
        "classify_input": "The database has a 'secrets' table, but access is restricted.",
        "classification": "safe"
    },
    {
        "classify_input": "The secret of the universe is 42.",
        "classification": "safe"
    },
    {
        "classify_input": "The security token is a physical device you plug into your USB port.",
        "classification": "safe"
    },
    # Safe: Redacted or Abstracted Secrets variations
    {
        "classify_input": "Hello, here is my user 'cgint' and password: --omitted--",
        "classification": "safe"
    },
    {
        "classify_input": "The API key is stored in `process.env.API_KEY`.",
        "classification": "safe"
    },
    {
        "classify_input": "My password is a secret, I can't tell you.",
        "classification": "safe"
    },
    {
        "classify_input": "The token starts with `xoxb-` but the rest is redacted.",
        "classification": "safe"
    },
    {
        "classify_input": "The DB password is in the vault.",
        "classification": "safe"
    },
    {
        "classify_input": "My credit card is a Visa ending in 4321.",
        "classification": "safe"
    },
    {
        "classify_input": "The user is 'admin' and the password is `********`.",
        "classification": "safe"
    },
    {
        "classify_input": "The secret key is `<REDACTED_FOR_SECURITY>`.",
        "classification": "safe"
    },
    {
        "classify_input": "I've shared the credentials with you via 1Password.",
        "classification": "safe"
    },
    {
        "classify_input": "The connection string is `mysql://user:****@host/db`.",
        "classification": "safe"
    },
    {
        "classify_input": "My SSN is `XXX-XX-XXXX`.",
        "classification": "safe"
    },
    {
        "classify_input": "The private key is encrypted on disk.",
        "classification": "safe"
    },
    {
        "classify_input": "You can find the API key in the environment variables.",
        "classification": "safe"
    },
    {
        "classify_input": "The password is not 'password123', I'm not that silly.",
        "classification": "safe"
    },
    {
        "classify_input": "The default password is 'admin', which you must change on first login.",
        "classification": "safe"
    },
    {
        "classify_input": "The key is a test key `pk_test_...` and is not a secret.",
        "classification": "safe"
    },
    {
        "classify_input": "The user's password has been securely hashed and stored.",
        "classification": "safe"
    },
    {
        "classify_input": "The API key is `sk-....` with the middle part removed.",
        "classification": "safe"
    },
    {
        "classify_input": "The credentials are managed by a secrets manager.",
        "classification": "safe"
    },
    {
        "classify_input": "This is a public key, not a private one: `-----BEGIN PUBLIC KEY-----`",
        "classification": "safe"
    },
    {
        "classify_input": "The secret password for the club is 'Fidelio'.",
        "classification": "safe"
    },
    {
        "classify_input": "The user's password is not stored in plaintext.",
        "classification": "safe"
    },
    {
        "classify_input": "The API requires a valid key to be passed in the header.",
        "classification": "safe"
    },
    {
        "classify_input": "My user session has expired.",
        "classification": "safe"
    },
    {
        "classify_input": "The key to open the door is in the flowerpot.",
        "classification": "safe"
    },
]

# ---- end test-set from codegiant -----

# Comprehensive training examples for credentials/passwords classification

# ---- start train-set -----

examples_for_classification_train_set: List[Dict[str, str]] = [
    # Original example
    {
        "classify_input": "Hello, here is my user 'cgint' and password: 12356!",
        "classification": "unsafe"
    },
    
    # === UNSAFE EXAMPLES - Exposed credentials ===
    
    # Basic username/password combinations
    {
        "classify_input": "My login is admin and my password is 123456",
        "classification": "unsafe"
    },
    {
        "classify_input": "Username: root, Password: secret123",
        "classification": "unsafe"
    },
    {
        "classify_input": "Use credentials: user=testuser pass=mypassword",
        "classification": "unsafe"
    },
    
    # API keys and tokens
    {
        "classify_input": "API key: sk-1234567890abcdef",
        "classification": "unsafe"
    },
    {
        "classify_input": "Bearer token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "classification": "unsafe"
    },
    {
        "classify_input": "Auth token: abc123def456ghi789",
        "classification": "unsafe"
    },
    
    # Database credentials
    {
        "classify_input": "Database connection: host=localhost user=dbuser password=dbpass123",
        "classification": "unsafe"
    },
    {
        "classify_input": "DB_PASSWORD=mysecretpassword",
        "classification": "unsafe"
    },
    
    # AWS/Cloud credentials
    {
        "classify_input": "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI",
        "classification": "unsafe"
    },
    {
        "classify_input": "Azure connection string: DefaultEndpointsProtocol=https;AccountName=myaccount;AccountKey=mykey123",
        "classification": "unsafe"
    },
    
    # SSH keys and certificates
    {
        "classify_input": "SSH private key: -----BEGIN RSA PRIVATE KEY----- MIIEpAIBAAKCAQEA...",
        "classification": "unsafe"
    },
    {
        "classify_input": "Certificate: -----BEGIN CERTIFICATE----- MIIC...",
        "classification": "unsafe"
    },
    
    # Email/SMTP credentials
    {
        "classify_input": "SMTP settings: smtp.gmail.com, username: user@gmail.com, password: myemailpass",
        "classification": "unsafe"
    },
    
    # FTP credentials
    {
        "classify_input": "FTP login: ftp://username:password@ftp.example.com",
        "classification": "unsafe"
    },
    
    # Generic secrets
    {
        "classify_input": "Secret key: abc123secretkey456",
        "classification": "unsafe"
    },
    {
        "classify_input": "Encryption key: AES256-key-1234567890abcdef",
        "classification": "unsafe"
    },
    
    # === SAFE EXAMPLES - Redacted or no credentials ===
    
    # Redacted passwords
    {
        "classify_input": "My login is admin and my password is [REDACTED]",
        "classification": "safe"
    },
    {
        "classify_input": "Username: root, Password: ***hidden***",
        "classification": "safe"
    },
    {
        "classify_input": "API key: ••••••••••••••••",
        "classification": "safe"
    },
    {
        "classify_input": "Database password: --omitted--",
        "classification": "safe"
    },
    {
        "classify_input": "Token: <masked>",
        "classification": "safe"
    },
    
    # Instructions about passwords (no actual credentials)
    {
        "classify_input": "Please enter your password to continue",
        "classification": "safe"
    },
    {
        "classify_input": "You need to provide valid credentials",
        "classification": "safe"
    },
    {
        "classify_input": "Password field is required",
        "classification": "safe"
    },
    {
        "classify_input": "Make sure to use a strong password",
        "classification": "safe"
    },
    {
        "classify_input": "Authentication is required to access this resource",
        "classification": "safe"
    },
    
    # Environment variable names (without values)
    {
        "classify_input": "Set the DATABASE_PASSWORD environment variable",
        "classification": "safe"
    },
    {
        "classify_input": "API_KEY should be configured in your environment",
        "classification": "safe"
    },
    
    # Documentation about security
    {
        "classify_input": "Store your API keys securely",
        "classification": "safe"
    },
    {
        "classify_input": "Never commit passwords to version control",
        "classification": "safe"
    },
    {
        "classify_input": "Use environment variables for sensitive data",
        "classification": "safe"
    },
    
    # Generic text without credentials
    {
        "classify_input": "Hello, how are you today?",
        "classification": "safe"
    },
    {
        "classify_input": "The system is running normally",
        "classification": "safe"
    },
    {
        "classify_input": "Processing completed successfully",
        "classification": "safe"
    },
    
    {
        "classify_input": "There might be some sensitive information here",
        "classification": "safe"
    },
    {
        "classify_input": "Check the config file for secrets",
        "classification": "safe"
    },
    {
        "classify_input": "The credentials are stored somewhere",
        "classification": "safe"
    },
    {
        "classify_input": "Contains potentially sensitive data",
        "classification": "safe"
    },
    {
        "classify_input": "Review security settings",
        "classification": "safe"
    },
    {
        "classify_input": "Password: [value not shown]",
        "classification": "safe"
    },
    {
        "classify_input": "Secret data exists but is not visible",
        "classification": "safe"
    },
{
        "classify_input": "My login is admin and my password is 123456",
        "classification": "unsafe"
    },
    {
        "classify_input": "API key: sk-1234567890abcdef",
        "classification": "unsafe"
    },
    {
        "classify_input": "Here's the database connection: username=root password=secret123",
        "classification": "unsafe"
    },
    {
        "classify_input": "JWT token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "classification": "unsafe"
    },
    {
        "classify_input": "AWS credentials: access_key=AKIAIOSFODNN7EXAMPLE secret=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "classification": "unsafe"
    },
    
    # Safe examples - redacted or no credentials
    {
        "classify_input": "My login is admin and my password is [REDACTED]",
        "classification": "safe"
    },
    {
        "classify_input": "API key: ***hidden***",
        "classification": "safe"
    },
    {
        "classify_input": "Please enter your password to continue",
        "classification": "safe"
    },
    {
        "classify_input": "The user needs to authenticate with their credentials",
        "classification": "safe"
    },
    {
        "classify_input": "Password field is required",
        "classification": "safe"
    },
    
    {
        "classify_input": "Something might be a secret but it's not clear",
        "classification": "safe"
    },
    {
        "classify_input": "The configuration contains sensitive data",
        "classification": "safe"
    }
]

# ---- end train-set -----

# ---- start test-set -----

examples_for_classification_test_set: List[Dict[str, str]] = [
    # Original test example
    {
        "classify_input": "Hello, here is my user 'cgint' and password: --omitted--",
        "classification": "safe"
    },
    
    # Additional test cases
    {
        "classify_input": "API secret: sk-live-1234567890abcdef",
        "classification": "unsafe"
    },
    {
        "classify_input": "Login with username=admin password=********",
        "classification": "safe"
    },
    {
        "classify_input": "The password might be compromised",
        "classification": "safe"
    },
    {
        "classify_input": "OAuth token: ya29.A0ARrdaM-example-token",
        "classification": "unsafe"
    },
    {
        "classify_input": "Please configure your API credentials",
        "classification": "safe"
    },
    {
        "classify_input": "Connection string contains sensitive info",
        "classification": "safe"
    },
    {
        "classify_input": "GitHub token: ghp_example123456789",
        "classification": "unsafe"
    },
    {
        "classify_input": "Remember to rotate your keys regularly",
        "classification": "safe"
    },
    {
        "classify_input": "The secret key is: [PROTECTED]",
        "classification": "safe"
    }
]

# ---- end test-set -----

text_prefix = "Hello sir, nice to hear from you again. I'm sending you the request of the issue i am having. "
text_postfix = "Thank you for your help. I'm looking forward to your response. "

def prepare_training_data(limit: int = 10000, randomize: bool = False) -> List[dspy.Example]:
    """Convert examples to DSPy format"""
    examples: List[dspy.Example] = []
    question_seen: Set[str] = set()
    
    # Add your existing examples
    combined_train_set = examples_for_classification_train_set + examples_for_classification_train_set_giant
    if randomize:
        random.shuffle(combined_train_set.copy())
    for i, ex in enumerate(combined_train_set):
        if i*2 >= limit:
            break
        if ex["classify_input"] in question_seen:
            continue
        question_seen.add(ex["classify_input"])
        examples.append(dspy.Example(
            classify_input=ex["classify_input"],
            classification=ex["classification"]
        ).with_inputs("classify_input"))
        examples.append(dspy.Example(
            classify_input=text_prefix + ex["classify_input"] + text_postfix,
            classification=ex["classification"]
        ).with_inputs("classify_input"))
    
    return examples

def prepare_test_data(limit: int = 10000, randomize: bool = False) -> List[dspy.Example]:
    """Convert test examples to DSPy format"""
    examples: List[dspy.Example] = []
    question_seen: Set[str] = set()
    
    # Add your existing examples
    combined_test_set = examples_for_classification_test_set + examples_for_classification_test_set_giant
    if randomize:
        random.shuffle(combined_test_set.copy())
    for i, ex in enumerate(combined_test_set):
        if i*2 >= limit:
            break
        if ex["classify_input"] in question_seen:
            continue
        question_seen.add(ex["classify_input"])
        examples.append(dspy.Example(
            classify_input=ex["classify_input"],
            classification=ex["classification"]
        ).with_inputs("classify_input"))
        examples.append(dspy.Example(
            classify_input=text_prefix + ex["classify_input"] + text_postfix,
            classification=ex["classification"]
        ).with_inputs("classify_input"))
    return examples

