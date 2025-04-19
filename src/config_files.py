import re

path_to_dataset = "/path/to/Dataset"
path_to_models = "/path/to/Models/"


caption_mapping = {
    "^.*: *": "",
    "contractor": "",
    ";": "",
    "\?": "",
    "it was found": "",
    "as seen in the video": "",
    "as seen in the image": "",
    "as shown in the photos": "",
    "[1-100][1-100]*": "",
    "fire fighter": "fire extinguisher",
    "fire-fighter": "fire extinguisher",
    "firefighter": "fire extinguisher",
    "Fire fighter": "fire extinguisher",
    "stowage": "storage",
    "i'm sorry": "",
    "I'm sorry": "",
    "I am sorry": "",
    "Am sorry": "",
    "im sorry": "",
    "i 'm sorry": "",
    "i' m sorry": "",
    "hurry up": "",
    "Hurry up": "",
    " *\. *\. *, *\. *\.": "",
    '"': "",
    "[0-9]": "",
    "\. \.": " ",
    "[Ii] do not know if you agree": "",
    "[iI] hope [iI] am clear": "",
    "[iI] think": "",
    "[oO]n the one hand": "",
    "[oO]n one hand": "",
    "[oO]n the other hand": "",
    "needs to be": "not",
    "should be": "not",
    "must be": "not",
    "fire fire": "fire",
    "fire fires": "fire",
    "fires fires": "fire",
    "work work": "work",
    "work works": "work",
    "works work": "work",
    "dusting, dusting, dusting": "dust",
    "the employee knows that he will not step on it anyway": "",
    "by a third party": "",
    "at times": "",
    "last year": "",
    "last semester": "",
    "last month": "",
    "concerns were raised": "",
    "in addition": "and",
    "in addition to": "and",
    "[bB]azza": "waste",
    "firehouse": "firechamber",
    "firefighters": "fire extinguishers",
    "firefighter": "fire extinguisher",
    "\(.*\)": "",  # remember to escape \(\) parenthesis since the regex in python uses the extended set of regex
}

# also we will add rules to delete the whole sample(image-caption pair) when they have only
# "" empty captions or only one symbol "." , "!" , ";" ,"-" ,"""
# captions only filled with symbols (.\/><'")
# captions with "as seen in the video"
# captions with "as seen in the image"

# add here the stopwords set we are going to use
stopwords_set = "custom_english"


# add strings to match on df['issue'] column
# and drop those rows
drop_patterns = [
    "as seen on the strip",
    "as seen on the outside.",
    "as seen on the outside",
    "as seen on the video.",
    "as seen on the image",
    "as seen",
    "as seen in the video.",
    "as seen in the video",
    "as seen on the image.",
    "as seen in the image.",
    "as seen in the image",
    "depot t : excluded p . and p . s .",
    "not found",
    "as attached",
    "does not exist.",
    "does not exist",
    "it does not exist",
    "it does not exist.",
    "as attached.",
    "as attached",
    "as seen on the strip",
    "as stated in the declaration.",
    "I'm sorry.",
    "i'm sorry",
    "i'm sorry.",
    "I am sorry",
    "I'm sorry",
    "I'm sorry,",
    "hurry up",
    "Hurry up",
    '" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "',
    "χ . θ ( a ) : 31 + 863 χ . θ ( b ) : 33 + 754",
    ".",
    " ",
    "",
    "as seen on the photo",
    '" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "',
    '" " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " " "',
    "without wines",
]
