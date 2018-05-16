"""Visualizing Twitter Sentiment Across America"""

from data import word_sentiments, load_tweets
from datetime import datetime
from geo import us_states, geo_distance, make_position, longitude, latitude
try:
    import tkinter
    from maps import draw_state, draw_name, draw_dot, wait
    HAS_TKINTER = True
except ImportError as e:
    print('Could not load tkinter: ' + str(e))
    HAS_TKINTER = False
from string import ascii_letters
from ucb import main, trace, interact, log_current_line


###################################
# Phase 1: The Feelings in Tweets #
###################################

# tweet data abstraction (A), represented as a list
# -------------------------------------------------

def make_tweet(text, time, lat, lon):
    """Return a tweet, represented as a Python list.

    Arguments:
    text  -- A string; the text of the tweet, all in lowercase
    time  -- A datetime object; the time that the tweet was posted
    lat   -- A number; the latitude of the tweet's location
    lon   -- A number; the longitude of the tweet's location

    >>> t = make_tweet('just ate lunch', datetime(2014, 9, 29, 13), 122, 37)
    >>> tweet_text(t)
    'just ate lunch'
    >>> tweet_time(t)
    datetime.datetime(2014, 9, 29, 13, 0)
    >>> p = tweet_location(t)
    >>> latitude(p)
    122
    >>> tweet_string(t)
    '"just ate lunch" @ (122, 37)'
    """
    return [text, time, lat, lon]

def tweet_text(tweet):
    """Return a string, the words in the text of a tweet."""
    "*** YOUR CODE HERE ***"
    return tweet[0]

def tweet_time(tweet):
    """Return the datetime representing when a tweet was posted."""
    "*** YOUR CODE HERE ***"
    return tweet[1]

def tweet_location(tweet):
    """Return a position representing a tweet's location."""
    "*** YOUR CODE HERE ***"
    return make_position(tweet[2] , tweet[3])

# tweet data abstraction (B), represented as a function
# -----------------------------------------------------

def make_tweet_fn(text, time, lat, lon):
    """An alternate implementation of make_tweet: a tweet is a function.

    >>> t = make_tweet_fn('just ate lunch', datetime(2014, 9, 29, 13), 122, 37)
    >>> tweet_text_fn(t)
    'just ate lunch'
    >>> tweet_time_fn(t)
    datetime.datetime(2014, 9, 29, 13, 0)
    >>> latitude(tweet_location_fn(t))
    122
    """
    # Please don't call make_tweet in your solution
    "*** YOUR CODE HERE ***"
    def tweet(x):
        if x == 'text':
            return text 
        elif x == 'time':
            return time
        elif x == 'lat':
            return lat
        else:
            return lon 
    return tweet 

def tweet_text_fn(tweet):
    """Return a string, the words in the text of a functional tweet."""
    return tweet('text')

def tweet_time_fn(tweet):
    """Return the datetime representing when a functional tweet was posted."""
    return tweet('time')

def tweet_location_fn(tweet):
    """Return a position representing a functional tweet's location."""
    return make_position(tweet('lat'), tweet('lon'))

### === +++ ABSTRACTION BARRIER +++ === ###

def tweet_string(tweet):
    """Return a string representing a tweet."""
    location = tweet_location(tweet)
    point = (latitude(location), longitude(location))
    return '"{0}" @ {1}'.format(tweet_text(tweet), point)

def tweet_words(tweet):
    """Return the words in a tweet."""
    return extract_words(tweet_text(tweet))

def extract_words(text):
    """Return the words in a tweet, not including punctuation.

    >>> extract_words('anything else.....not my job')
    ['anything', 'else', 'not', 'my', 'job']
    >>> extract_words('i love my job. #winning')
    ['i', 'love', 'my', 'job', 'winning']
    >>> extract_words('make justin # 1 by tweeting #vma #justinbieber :)')
    ['make', 'justin', 'by', 'tweeting', 'vma', 'justinbieber']
    >>> extract_words("paperclips! they're so awesome, cool, & useful!")
    ['paperclips', 'they', 're', 'so', 'awesome', 'cool', 'useful']
    >>> extract_words('@(cat$.on^#$my&@keyboard***@#*')
    ['cat', 'on', 'my', 'keyboard']
    """
    "*** YOUR CODE HERE ***"
    extracted_words = []
    index = 0
    for word in text:
        if text[index] in ascii_letters:
            extracted_words.append(text[index])
        else:
            extracted_words.append(' ')
        index += 1
    text = ''.join(extracted_words)
    return text.split()  # You may change/remove this line

def make_sentiment(value):
    """Return a sentiment, which represents a value that may not exist.

    >>> positive = make_sentiment(0.2)
    >>> neutral = make_sentiment(0)
    >>> unknown = make_sentiment(None)
    >>> has_sentiment(positive)
    True
    >>> has_sentiment(neutral)
    True
    >>> has_sentiment(unknown)
    False
    >>> sentiment_value(positive)
    0.2
    >>> sentiment_value(neutral)
    0
    """
    assert (value is None) or (-1 <= value <= 1), 'Bad sentiment value'
    "*** YOUR CODE HERE ***"
    if value != None:
        return value
    else:
        return None

def has_sentiment(s):
    """Return whether sentiment s has a value."""
    "*** YOUR CODE HERE ***"
    if s != None:
        return True
    else:
        return False

def sentiment_value(s):
    """Return the value of a sentiment s."""
    assert has_sentiment(s), 'No sentiment value'
    "*** YOUR CODE HERE ***"
    return s

def get_word_sentiment(word):
    """Return a sentiment representing the degree of positive or negative
    feeling in the given word.

    >>> sentiment_value(get_word_sentiment('good'))
    0.875
    >>> sentiment_value(get_word_sentiment('bad'))
    -0.625
    >>> sentiment_value(get_word_sentiment('winning'))
    0.5
    >>> has_sentiment(get_word_sentiment('Berkeley'))
    False
    """
    # Learn more: http://docs.python.org/3/library/stdtypes.html#dict.get
    return make_sentiment(word_sentiments.get(word))

def analyze_tweet_sentiment(tweet):
    """Return a sentiment representing the degree of positive or negative
    feeling in the given tweet, averaging over all the words in the tweet
    that have a sentiment value.

    If no words in the tweet have a sentiment value, return
    make_sentiment(None).

    >>> positive = make_tweet('i love my job. #winning', None, 0, 0)
    >>> round(sentiment_value(analyze_tweet_sentiment(positive)), 5)
    0.29167
    >>> negative = make_tweet("saying, 'i hate my job'", None, 0, 0)
    >>> sentiment_value(analyze_tweet_sentiment(negative))
    -0.25
    >>> no_sentiment = make_tweet('berkeley golden bears!', None, 0, 0)
    >>> has_sentiment(analyze_tweet_sentiment(no_sentiment))
    False
    """
    "*** YOUR CODE HERE ***"
    word_list = tweet_words(tweet)
    total = 0
    num_words = 0
    for word in word_list:
        if has_sentiment(get_word_sentiment(word)):
            total += sentiment_value(get_word_sentiment(word))
            num_words += 1
        else:
            total += 0
    if num_words == 0:
        return make_sentiment(None)
    else:
        return make_sentiment(total / num_words)


#################################
# Phase 2: The Geometry of Maps #
#################################

def apply_to_all(map_fn, s):
    return [map_fn(x) for x in s]

def keep_if(filter_fn, s):
    return [x for x in s if filter_fn(x)]

def find_centroid(polygon):
    """Find the centroid of a polygon. If a polygon has 0 area, use the latitude
    and longitude of its first position as its centroid.

    http://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon

    Arguments:
    polygon -- A list of positions, in which the first and last are the same

    Returns 3 numbers: centroid latitude, centroid longitude, and polygon area.

    >>> p1 = make_position(1, 2)
    >>> p2 = make_position(3, 4)
    >>> p3 = make_position(5, 0)
    >>> triangle = [p1, p2, p3, p1] # First vertex is also the last vertex
    >>> round_all = lambda s: [round(x, 5) for x in s]
    >>> round_all(find_centroid(triangle))
    [3.0, 2.0, 6.0]
    >>> round_all(find_centroid([p1, p3, p2, p1])) # reversed
    [3.0, 2.0, 6.0]
    >>> apply_to_all(float, find_centroid([p1, p2, p1])) # A zero-area polygon
    [1.0, 2.0, 0.0]
    """
    "*** YOUR CODE HERE ***"
    # A, Cx, Cy = 0, 0, 0
    # for i in range(0, len(polygon) - 1):
    #     A += polygon[i](latitude(i))*polygon[i+1](longitude(i+1)) - polygon[i+1](latitude(i+1))*polygon[i](longitude(i))
    #     Cx += (polygon[i](latitude(i)) + polygon[i+1](latitude(i+1)))*(polygon[i](latitude(i))*polygon[i+1](longitude(i+1)) - polygon[i+1](latitude(i+1))*polygon[i](longitude(i)))
    #     Cy += (polygon[i](longitude(i)) + polygon[i+1](longitude(i+1)))*(polygon[i](latitude(i))*polygon[i+1](longitude(i+1)) - polygon[i+1](latitude(i+1))*polygon[i](longitude(i)))
    # A = A / 2
    # if A != 0:
    #     Cx = Cx / (6*A)
    #     Cy = Cy / (6*A)
    # else:
    #     Cx = polygon[0][0]
    #     Cy = polygon[0][1]
    # if A < 0:
    #     A = abs(A)
    #     Cx = abs(Cx)
    #     Cy = abs(Cy)
    # return Cx, Cy, A
    lats = [latitude(position) for position in polygon]
    longs = [longitude(position) for position in polygon]
    pre_A = [lats[i] * longs[i + 1] - lats[i + 1] * longs[i] for i in range(len(polygon) - 1)]
    A = 0.5 * sum(pre_A)

    if A == 0.0:
        return [latitude(polygon[0]), longitude(polygon[0]), A]
    else:
        pre_Cx = [(lats[i] + lats[i + 1]) * (lats[i] * longs[i + 1] - lats[i + 1] * longs[i]) for i in range(len(polygon) - 1)]
        Cx = (1 / (6 * A)) * sum(pre_Cx)

        pre_Cy = [(longs[i] + longs[i + 1]) * (lats[i] * longs[i + 1] - lats[i + 1] * longs[i]) for i in range(len(polygon) - 1)]
        Cy = (1 / (6 * A)) * sum(pre_Cy)

        return [Cx, Cy, abs(A)]

def find_state_center(polygons):
    """Compute the geographic center of a state, averaged over its polygons.

    The center is the average position of centroids of the polygons in
    polygons, weighted by the area of those polygons.

    Arguments:
    polygons -- a list of polygons

    >>> ca = find_state_center(us_states['CA'])  # California
    >>> round(latitude(ca), 5)
    37.25389
    >>> round(longitude(ca), 5)
    -119.61439

    >>> hi = find_state_center(us_states['HI'])  # Hawaii
    >>> round(latitude(hi), 5)
    20.1489
    >>> round(longitude(hi), 5)
    -156.21763
    """
    "*** YOUR CODE HERE ***"
    list_of_c_x = [find_centroid(polygons[i])[0] for i in range(len(polygons))]
    list_of_c_y = [find_centroid(polygons[i])[1] for i in range(len(polygons))]
    list_of_areas = [find_centroid(polygons[i])[2] for i in range(len(polygons))]
    
    average_c_x_prelim_list = [list_of_c_x[i] * list_of_areas[i] for i in range(len(polygons))]
    average_c_y_prelim_list = [list_of_c_y[i] * list_of_areas[i] for i in range(len(polygons))]

    average_c_x = sum(average_c_x_prelim_list) / sum(list_of_areas)
    average_c_y = sum(average_c_y_prelim_list) / sum(list_of_areas)

    return make_position(average_c_x, average_c_y)

    # lat, lon = 0, 0
    # index = 0 
    # for polygon in polygons:
    #     if type(polygon[index]) != list: 
    #         Cx, Cy, A = find_centroid(polygon)
    #         lat += Cx*A 
    #         lon += Cy*A
    #     if type(polygon[index]) == list:
    #         find_centroid(polygon[index])
    #     index += 1
    # Cx = lat/A
    # Cy = lon/A 
    # return Cx, Cy 

    # lat, lon = 0, 0
    # for i in range(0, len(polygons)):
    #     Cx, Cy, A = find_centroid(polygons[i])
    #     lat += Cx*A
    #     lon += Cy*A
    # Cx = lat/A
    # Cy = lon/A
    # return (Cx, Cy) 

    # number_of_polygon = len(polygons)
    # A, CxA, CyA = 0, 0, 0
    # for i in range(0, number_of_polygon):
    #     centroid_area = find_centroid(polygons[i])
    #     CxA += centroid_area[0] * centroid_area[2]
    #     CyA += centroid_area[1] * centroid_area[2]
    #     A += centroid_area[2]
    # CxA = CxA / A
    # CyA = CyA / A
    # return CxA, CyA

    # centroid_area = [find_centroid(polygon) for polygon in polygons]
    # CixA = [lat_lon_area[0][0]*lat_lon_area[0][2] for lat_lon_area in centroid_area]
    # CiyA = [lat_lon_area[0][1]*lat_lon_area[0][2] for lat_lon_area in centroid_area]
    # A = [lat_lon_area[0][2] for lat_lon_area in centroid_area]

    # Cx = sum(CixA) / sum(A)
    # Cy  = sum(CiyA) / sum(A)
    # return [Cx, Cy]




###################################
# Phase 3: The Mood of the Nation #
###################################

def group_by_key(pairs):
    """Return a dictionary that relates each unique key in [key, value] pairs
    to a list of all values that appear paired with that key.

    Arguments:
    pairs -- a sequence of pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_key(example)
    {1: [2, 3, 2], 2: [4], 3: [2, 1]}
    """
    # Optional: This implementation is slow because it traverses the list of
    #           pairs one time for each key. Can you improve it?
    keys = [key for key, _ in pairs]
    return {key: [y for x, y in pairs if x == key] for key in keys}

def group_tweets_by_state(tweets):
    """Return a dictionary that groups tweets by their nearest state center.

    The keys of the returned dictionary are state names and the values are
    lists of tweets that appear closer to that state center than any other.

    Arguments:
    tweets -- a sequence of tweet abstract data types

    >>> sf = make_tweet("welcome to san francisco", None, 38, -122)
    >>> ny = make_tweet("welcome to new york", None, 41, -74)
    >>> two_tweets_by_state = group_tweets_by_state([sf, ny])
    >>> len(two_tweets_by_state)
    2
    >>> california_tweets = two_tweets_by_state['CA']
    >>> len(california_tweets)
    1
    >>> tweet_string(california_tweets[0])
    '"welcome to san francisco" @ (38, -122)'
    """
    "*** YOUR CODE HERE ***"
    state_centers = { key: find_state_center(us_states[key]) for key in us_states } # Dictionary which maps each state code to its center position.

    lst_of_closest_states = []
    for tweet in tweets:
        distance_from_each_state = { key: geo_distance(tweet_location(tweet), state_centers[key]) for key in state_centers } # A dictionary which maps each state to its distance from the location of a tweet.
        closest_state = min(distance_from_each_state, key = distance_from_each_state.get)
        lst_of_closest_states.append([closest_state, tweet]) 
    # lst_of_closest_states contains sequences of state codes and tweets whose locations are closest to said state
    
    return group_by_key(lst_of_closest_states) # This gives a dictionary which maps state codes to a list of tweets whose locations are closest to said state

    # build state list
    # get list of polygons from us_states
    # find state center of each state
    # build dictionary of states with its center
    # compare state centers with lat/lon of tweet using geo_distance
    # try using min
    # group_by_key

    # state = []
    # polygons = []
    # for s, p in us_states:
    #     state.append(s)
    #     position0 = find_state_center(p)
    #     position1 = tweet_location(tweet)
    #     polygons.append(geo_distance(position0, position1))
    # return min(polygons)

    # state = []
    # polygons = []
    # for s, p in us_states.items(): # create list of states and list of polygons
    #     state.append(s)
    #     polygons.append(p)
    # centers_of_states = []
    # for p in polygons: # create list of state centers
    #     centers = find_state_center(p)
    #     centers_of_states.append(centers)
    # number_of_tweets = len(tweets)
    # distances = []
    # for i in range(0, number_of_tweets):
    #     st_tweet = make_tweet(tweets[i][0], tweets[i][1], tweets[i][2], tweets[i][3])
    #     X, Y = tweet_location(st_tweet)
    #     for center in centers_of_states:
    #         state_center_position = center
    #         tweet_position = make_position(X, Y)
    #         distances.append(geo_distance(tweet_position, state_center_position))
    # return min(distances)




        # position0 = find_state_center(polygons)
        # position1 = tweet_location(tweet)
        # geo_distance(position0, position1)
        # us_states[state] = # add min geodistance of each tweet to dicitonary of states




def average_sentiments(tweets_by_state):
    """Calculate the average sentiment of the states by averaging over all
    the tweets from each state. Return the result as a dictionary from state
    names to average sentiment values (numbers).

    If a state has no tweets with sentiment values, leave it out of the
    dictionary entirely. Do NOT include states with no tweets, or with tweets
    that have no sentiment, as 0. 0 represents neutral sentiment, not unknown
    sentiment.

    Arguments:
    tweets_by_state -- A dictionary from state names to lists of tweets
    """
    "*** YOUR CODE HERE ***"
    # gets average sentiment value of entire state
    # returns dictionary of state name and average sentiment value

    def average(list_of_tweets):
        tweets_with_sentiment = [ sentiment_value(analyze_tweet_sentiment(tweet)) for tweet in list_of_tweets if has_sentiment(analyze_tweet_sentiment(tweet)) ]
        if tweets_with_sentiment == []:
            return None
        else:
            return sum(tweets_with_sentiment) / len(tweets_with_sentiment)

    return { key: average(tweets_by_state[key]) for key in tweets_by_state if average(tweets_by_state[key]) != None }

##########################
# Command Line Interface #
##########################

def uses_tkinter(func):
    """A decorator that designates a function as one that uses tkinter.
    If tkinter is not supported, will not allow these functions to run.
    """
    def tkinter_checked(*args, **kwargs):
        if HAS_TKINTER:
            return func(*args, **kwargs)
        print('tkinter not supported, cannot call {0}'.format(func.__name__))
    return tkinter_checked

def print_sentiment(text='Are you virtuous or verminous?'):
    """Print the words in text, annotated by their sentiment scores."""
    words = extract_words(text.lower())
    layout = '{0:>' + str(len(max(words, key=len))) + '}: {1:+}'
    for word in words:
        s = get_word_sentiment(word)
        if has_sentiment(s):
            print(layout.format(word, sentiment_value(s)))

@uses_tkinter
def draw_centered_map(center_state='TX', n=10):
    """Draw the n states closest to center_state."""
    centers = {name: find_state_center(us_states[name]) for name in us_states}
    center = centers[center_state.upper()]
    distance = lambda name: geo_distance(center, centers[name])
    for name in sorted(centers, key=distance)[:int(n)]:
        draw_state(us_states[name])
        draw_name(name, centers[name])
    draw_dot(center, 1, 10)  # Mark the center state with a red dot
    wait()

@uses_tkinter
def draw_state_sentiments(state_sentiments):
    """Draw all U.S. states in colors corresponding to their sentiment value.

    Unknown state names are ignored; states without values are colored grey.

    Arguments:
    state_sentiments -- A dictionary from state strings to sentiment values
    """
    for name, shapes in us_states.items():
        draw_state(shapes, state_sentiments.get(name))
    for name, shapes in us_states.items():
        center = find_state_center(shapes)
        if center is not None:
            draw_name(name, center)

@uses_tkinter
def draw_map_for_query(term='my job', file_name='tweets2014.txt'):
    """Draw the sentiment map corresponding to the tweets that contain term.

    Some term suggestions:
    New York, Texas, sandwich, my life, justinbieber
    """
    tweets = load_tweets(make_tweet, term, file_name)
    tweets_by_state = group_tweets_by_state(tweets)
    state_sentiments = average_sentiments(tweets_by_state)
    draw_state_sentiments(state_sentiments)
    for tweet in tweets:
        s = analyze_tweet_sentiment(tweet)
        if has_sentiment(s):
            draw_dot(tweet_location(tweet), sentiment_value(s))
    wait()

def swap_tweet_representation(other=[make_tweet_fn, tweet_text_fn,
                                     tweet_time_fn, tweet_location_fn]):
    """Swap to another representation of tweets. Call again to swap back."""
    global make_tweet, tweet_text, tweet_time, tweet_location
    swap_to = tuple(other)
    other[:] = [make_tweet, tweet_text, tweet_time, tweet_location]
    make_tweet, tweet_text, tweet_time, tweet_location = swap_to




@main
def run(*args):
    """Read command-line arguments and calls corresponding functions."""
    import argparse
    parser = argparse.ArgumentParser(description="Run Trends")
    parser.add_argument('--print_sentiment', '-p', action='store_true')
    parser.add_argument('--draw_centered_map', '-d', action='store_true')
    parser.add_argument('--draw_map_for_query', '-m', type=str)
    parser.add_argument('--tweets_file', '-t', type=str, default='tweets2014.txt')
    parser.add_argument('--use_functional_tweets', '-f', action='store_true')
    parser.add_argument('text', metavar='T', type=str, nargs='*',
                        help='Text to process')
    args = parser.parse_args()
    if args.use_functional_tweets:
        swap_tweet_representation()
        print("Now using a functional representation of tweets!")
        args.use_functional_tweets = False
    if args.draw_map_for_query:
        print("Using", args.tweets_file)
        draw_map_for_query(args.draw_map_for_query, args.tweets_file)
        return
    for name, execute in args.__dict__.items():
        if name != 'text' and name != 'tweets_file' and execute:
            globals()[name](' '.join(args.text))
