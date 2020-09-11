update yelp_reviews_filtered
set text = replace(lower(text), 'one star', '');
commit;
