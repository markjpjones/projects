-- set review length attribute
update yelp_reviews_filtered
set review_length = array_length(regexp_split_to_array(trim("text"), E'\\s+'), 1);
commit;


-- remove reviews whose length is less than five
delete
from yelp_reviews_filtered
where length(text) < 5;
commit;

-- delete reviews that do not contain alphabetical characters
delete
from yelp_reviews_filtered
where text !~* '[A-Za-z]\w+';
commit;
