import tldextract

# Enhanced trusted domains (your original + more)
TRUSTED_DOMAINS = {
    # 🌍 International Mainstream News
    "abcnews.go.com", "aljazeera.com", "apnews.com", "bbc.com",
    "bloomberg.com", "cbc.ca", "cbsnews.com", "cnn.com",
    "dw.com", "economist.com", "euronews.com", "forbes.com",
    "ft.com", "indiatimes.com", "japantimes.co.jp", "latimes.com",
    "npr.org", "nytimes.com", "reuters.com", "smh.com.au",
    "theguardian.com", "usatoday.com", "washingtonpost.com", "wsj.com",
    "france24.com",

    # 📰 Ghana-Specific News
    "3news.com", "adomonline.com", "citinewsroom.com", "ghanaweb.com",
    "ghanaiantimes.com.gh", "ghananewsagency.org",
    "graphic.com.gh", "modernghana.com", "myjoyonline.com",
    "peacefmonline.com", "pulse.com.gh", "starrfm.com.gh", "thebftonline.com",
    "yen.com.gh",

    # ⚽ Sports News
    "cbssports.com", "espn.com", "eurosport.com", "fifa.com",
    "footballghana.com", "foxsports.com", "ghanasoccernet.com",
    "goal.com", "nba.com", "nbcsports.com", "onefootball.com",
    "skysports.com", "sportinglife.com", "supersport.com",
    "tntsports.co.uk", "theathletic.com", "olympics.com",

    # 🎬 Entertainment & Pop Culture
    "billboard.com", "deadline.com", "entertainment.com", "eonline.com",
    "ew.com", "hollywoodreporter.com", "indiewire.com", "people.com",
    "rollingstone.com", "thewrap.com", "variety.com",

    # 🧪 Science & Research
    "eurekalert.org", "medpagetoday.com", "nasa.gov", "nature.com",
    "sciencealert.com", "sciencenews.org", "statnews.com",

    # 🌐 Fact-Checking & Watchdogs
    "africacheck.org", "factcheck.org", "fullfact.org",
    "politifact.com", "snopes.com",

    # 🌍 Global & General Niche News
    "asia.nikkei.com", "globalissues.org", "ipsnews.net",
    "oecdobserver.org", "rferl.org",

    # 📰 African Regional News (non-Ghana)
    "dailynation.africa", "enca.com", "ewn.co.za",
    "monitor.co.ug", "thecitizen.co.tz", "businessinsider.com", 
    "africanews.com",

    # 🎓 Academic & Policy Think Tanks
    "brookings.edu", "carnegieendowment.org", "cfr.org",
    "foreignpolicy.com", "theconversation.com",
}

# Suspicious domains that often spread misinformation
SUSPICIOUS_DOMAINS = {
    "beforeitsnews.com", "naturalnews.com", "infowars.com",
    "breitbart.com", "dailystormer.com", "zerohedge.com",
    "activistpost.com", "realfarmacy.com", "healthnutnews.com",
}

def extract_domain(url):
    """Extract domain from URL"""
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"