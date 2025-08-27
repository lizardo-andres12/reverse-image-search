create table if not exists images (
    uuid UUID primary key not null,
    filename varchar(255) not null,
    source_url text not null,
    source_domain text not null,
    file_size integer not null,
    dimensions varchar(20) not null,
    created_at timestamp default NOW(),
    indexed_at timestamp
);

create table if not exists image_tags (
    id serial primary key
    image_uuid UUID not null references images(uuid) on delete cascade,
    tag varchar(63) not null,
    confidence float not null,
    unique(image_uuid, tag)
);

create index idx_image_tags_uuid on image_tags(image_uuid);
create index idx_image_tags_taf on image_tags(tag);
