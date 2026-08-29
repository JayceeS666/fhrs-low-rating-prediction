"""
Just the lookup tables — which local authorities count as England, and the
handful of places FHRS and IMD25 spell differently. 
"""

# using semicolons instead of commas here because one of the names
# (Bournemouth, Christchurch and Poole) has a comma in it
_REGIONS_RAW = {
    "East Counties": (
        "Babergh;Basildon;Bedford;Braintree;Breckland;Brentwood;Broadland;"
        "Broxbourne;Cambridge City;Castle Point;Central Bedfordshire;Chelmsford;"
        "Colchester;Dacorum;East Cambridgeshire;East Hertfordshire;East Suffolk;"
        "Epping Forest;Fenland;Great Yarmouth;Harlow;Hertsmere;Huntingdonshire;"
        "Ipswich;King's Lynn and West Norfolk;Luton;Maldon;Mid Suffolk;"
        "North Hertfordshire;North Norfolk;Norwich City;Peterborough City;"
        "Rochford;South Cambridgeshire;South Norfolk;Southend-On-Sea;"
        "St Albans City;Stevenage;Tendring;Three Rivers;Thurrock;Uttlesford;"
        "Watford;Welwyn Hatfield;West Suffolk"
    ),
    "East Midlands": (
        "Amber Valley;Ashfield;Bassetlaw;Blaby;Bolsover;Boston;Broxtowe;"
        "Charnwood;Chesterfield;Derby City;Derbyshire Dales;East Lindsey;Erewash;"
        "Gedling;Harborough;High Peak;Hinckley and Bosworth;Leicester City;"
        "Lincoln City;Mansfield;Melton;Newark and Sherwood;North East Derbyshire;"
        "North Kesteven;North Northamptonshire;North West Leicestershire;"
        "Nottingham City;Oadby and Wigston;Rushcliffe;Rutland;South Derbyshire;"
        "South Holland;South Kesteven;West Lindsey;West Northamptonshire"
    ),
    "London": (
        "Barking and Dagenham;Barnet;Bexley;Brent;Bromley;Camden;"
        "City of London Corporation;Croydon;Ealing;Enfield;Greenwich;Hackney;"
        "Hammersmith and Fulham;Haringey;Harrow;Havering;Hillingdon;Hounslow;"
        "Islington;Kensington and Chelsea;Kingston-Upon-Thames;Lambeth;Lewisham;"
        "Merton;Newham;Redbridge;Richmond-Upon-Thames;Southwark;Sutton;"
        "Tower Hamlets;Waltham Forest;Wandsworth;Westminster"
    ),
    "North East": (
        "Darlington;Durham;Gateshead;Hartlepool;Middlesbrough;"
        "Newcastle Upon Tyne;North Tyneside;Northumberland;Redcar and Cleveland;"
        "River Tees;South Tyneside;Stockton On Tees;Sunderland"
    ),
    "North West": (
        "Blackburn;Blackpool;Bolton;Burnley;Bury;Cheshire East;"
        "Cheshire West and Chester;Chorley;Cumberland;Fylde;Halton;Hyndburn;"
        "Knowsley;Lancaster City;Liverpool;Manchester;Oldham;Pendle;Preston;"
        "Ribble Valley;Rochdale;Rossendale;Salford;Sefton;South Ribble;St Helens;"
        "Stockport;Tameside;Trafford;Warrington;West Lancashire;"
        "Westmorland and Furness;Wigan;Wirral;Wyre"
    ),
    "South East": (
        "Adur;Arun;Ashford;Basingstoke and Deane;Bracknell Forest;"
        "Brighton and Hove;Buckinghamshire;Canterbury City;Cherwell;Chichester;"
        "Crawley;Dartford;Dover;East Hampshire;Eastbourne;Eastleigh;Elmbridge;"
        "Epsom and Ewell;Fareham;Folkestone and Hythe;Gosport;Gravesham;"
        "Guildford;Hart;Hastings;Havant;Horsham;Isle of Wight;Lewes;Maidstone;"
        "Medway;Mid Sussex;Milton Keynes;Mole Valley;New Forest;Oxford City;"
        "Portsmouth;Reading;Reigate and Banstead;Rother;Runnymede;Rushmoor;"
        "Sevenoaks;Slough;South Oxfordshire;Southampton;Spelthorne;Surrey Heath;"
        "Swale;Tandridge;Test Valley;Thanet;Tonbridge and Malling;"
        "Tunbridge Wells;Vale of White Horse;Waverley;Wealden;West Berkshire;"
        "West Oxfordshire;Winchester City;Windsor and Maidenhead;Woking;"
        "Wokingham;Worthing"
    ),
    "South West": (
        "Bath and North East Somerset;Bournemouth, Christchurch and Poole;"
        "Bristol;Cheltenham;Cornwall;Cotswold;Dorset;East Devon;Exeter City;"
        "Forest of Dean;Gloucester City;Isles of Scilly;Mid Devon;North Devon;"
        "North Somerset;Plymouth City;Somerset;South Gloucestershire;South Hams;"
        "Stroud;Swindon;Teignbridge;Tewkesbury;Torbay;Torridge;West Devon;"
        "Wiltshire"
    ),
    "West Midlands": (
        "Birmingham;Bromsgrove;Cannock Chase;Coventry;Dudley;East Staffordshire;"
        "Herefordshire;Lichfield;Malvern Hills;Newcastle-Under-Lyme;"
        "North Warwickshire;Nuneaton and Bedworth;Redditch;Rugby;Sandwell;"
        "Shropshire;Solihull;South Staffordshire;Stafford;"
        "Staffordshire Moorlands;Stoke-On-Trent;Stratford-on-Avon;Tamworth;"
        "Telford and Wrekin Council;Walsall;Warwick;Wolverhampton;Worcester City;"
        "Wychavon;Wyre Forest"
    ),
    "Yorkshire and Humberside": (
        "Barnsley;Bradford;Calderdale;Doncaster;East Riding of Yorkshire;"
        "Hull and Goole Port;Hull City;Kirklees;Leeds;North East Lincolnshire;"
        "North Lincolnshire;North Yorkshire;Rotherham;Sheffield;Wakefield;York"
    ),
}

ENGLAND_REGIONS = {r: s.split(";") for r, s in _REGIONS_RAW.items()}

# FHRS and IMD25 don't always use the same name for the same place
IMD_NAME_FIX = {
    "Blackburn with Darwen": "Blackburn",
    "Bristol, City of": "Bristol",
    "Cambridge": "Cambridge City",
    "Canterbury": "Canterbury City",
    "City of London": "City of London Corporation",
    "County Durham": "Durham",
    "Derby": "Derby City",
    "Exeter": "Exeter City",
    "Gloucester": "Gloucester City",
    "Herefordshire, County of": "Herefordshire",
    "Kingston upon Hull, City of": "Hull City",
    "Kingston upon Thames": "Kingston-Upon-Thames",
    "Lancaster": "Lancaster City",
    "Leicester": "Leicester City",
    "Lincoln": "Lincoln City",
    "Newcastle upon Tyne": "Newcastle Upon Tyne",
    "Newcastle-under-Lyme": "Newcastle-Under-Lyme",
    "Norwich": "Norwich City",
    "Nottingham": "Nottingham City",
    "Oxford": "Oxford City",
    "Peterborough": "Peterborough City",
    "Plymouth": "Plymouth City",
    "Richmond upon Thames": "Richmond-Upon-Thames",
    "Southend-on-Sea": "Southend-On-Sea",
    "St Albans": "St Albans City",
    "St. Helens": "St Helens",
    "Stockton-on-Tees": "Stockton On Tees",
    "Stoke-on-Trent": "Stoke-On-Trent",
    "Telford and Wrekin": "Telford and Wrekin Council",
    "Winchester": "Winchester City",
    "Worcester": "Worcester City",
}
