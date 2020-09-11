-- create filtered business table to only include restaurants
create table yelp_business_filtered as
select *
from yelp_business
where categories ilike '%restaurant%';
commit;

-- filter reviews to only include restaurant reviews
delete
from yelp_reviews_filtered
where business_id not in (select business_id from yelp_business_filtered);
commit;
