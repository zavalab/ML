# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:56:31 2021

@author: sqin34
"""


# https://github.com/oscarcontrerasnavas/NIST-web-book-scraping/blob/master/saturation.py
# Import get function from requests module because is the function in charge of
# getting the HTTP GET request with the given url.
from requests import get

# Import BeautifulSoup from bs4 because make the html parse and help us to
# handle de DOM.
from bs4 import BeautifulSoup

# Import closing for ensure that any network resource will free when they go out
# of scope.
from contextlib import closing

def get_antoine_coef(Name, Temperature):

    """ Return a list with the coefficients A, B and C if they exist for the
        given Temperature. If not, return None and print it.
    :param Name:
        A string with the name of the compound in English.
    :param Temperature:
        A float number with the temperature in Kelvin.
    :rtype: List
    :return coef with [A, B, C]
    """

    # Obtaining the table using the get_html function showed below. Table is a
    # BeautifulSoup Object.
    table = get_html_table(Name)

    # Extract the rows from the table. Knowing what tags have an HTML table.
    # Also, knowing that the fist row with he table header does not have the
    # class attribute 'exp' so we obtain just the rows with data.
    # The find_all function from BeautifulSoup return a list
    rows = table.find_all('tr', class_='exp')

    # Declaring the lists for storage Temperatures, and coefficients.
    Temperatures, As, Bs, Cs = [], [], [], []

    # Looping over rows to extract and fill As, Bs, and Cs variables because now
    # we are sure the Temperatues is between some range.
    for row in rows:

        # As the rows, we extract the columns for the current row. Knowing that
        # the cols have the <td> tag in HTML as well
        # The find_all function from BeautifulSoup return a list
        cols =  row.find_all('td')

        # First transform the strings into float numbers and put them in their
        # respective list
        As.append(float(cols[1].text))
        Bs.append(float(cols[2].text))
        Cs.append(float(cols[3].text))

        # For the temperatures, we have a range and we need to extract each
        # limit (lower and higher) and put them in an extra list. So
        # Temperatures variable will be a list of lists.
        lower_lim = float(cols[0].text.replace(" ","").split('-')[0])
        higher_lim = float(cols[0].text.replace(" ","").split('-')[1])
        Temperatures.append([lower_lim, higher_lim])


    # Checking if the Temperature gave fits in some interval
    index = None
    for i, interval in enumerate(Temperatures):
        if (interval[0] <= Temperature
            and Temperature <= interval[1]):
            index = i
            break
        else:
            index = None

    if index == None:
        raise ValueError('Sorry, the data for the given temperature %.2f K does not exist in the Data Base' % Temperature)
    else:
        A = As[index]
        B = Bs[index]
        C = Cs[index]
        return [A, B, C]


def get_html_table(Name):

    """ Return the html already parsed using the a helper function listed below.
    :param Name:
        A string with the name of the compound in English.
    :rtype: BeautifulSoup Object
    """

    # The name parameter is part of the url. For example, if you want the
    # methane data, the url is
    # https://webbook.nist.gov/cgi/cbook.cgi?Name=methane&Mask=4.
    url = str.format('https://webbook.nist.gov/cgi/cbook.cgi?Name={0}&Mask=4', Name.lower())

    # Function to get the request made, see below.
    raw_html = get_response(url)

    # Parse the html using BeautifulSoup.
    html = BeautifulSoup(raw_html, 'html.parser')

    # Extract the table that contains the data, the table has a specific
    # attributes 'aria-label' as 'Antoine Equation Parameters'.
    table = html.find('table', attrs={'aria-label': 'Antoine Equation Parameters'})

    return table


def get_response(url):

    """ Return the raw_html for parsing later or None if can't reach the page
    :param url:
        The string for the GET request.
    :rtype: BeautifulSoup Object
    :rtype: None if can't reach the website
    """

    try: 
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None
    
    except:
        print('Not found')
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None 
            and content_type.find('html') > -1)        
