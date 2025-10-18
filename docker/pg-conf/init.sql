create table if not exists images (
    uuid UUID primary key not null,
    filename varchar(255) not null,
    source_url text not null,
    source_domain text not null,
    file_size integer not null,
    dimensions varchar(20) not null,
    created_at timestamp default NOW(),
    indexed_at timestamp default NOW()
);
