import scrapy


class BlogSpider(scrapy.Spider):
    name = 'narutopider'
    start_urls = [
        'https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu?limit=250&offset=0&_cat=Jutsu']

    def parse(self, response):
        for href in response.css('div.smw-columnlist-container')[0].css('a::attr(href)').extract():
            extracted_data = scrapy.Request("https://naruto.fandom.com"+href,
                                            callback=self.parse_jutsu)
            return extracted_data

        for next_page in response.css('a.mw-nextlink'):
            return response.follow(next_page, self.parse)

    def parse_jutsu(self, response):
        jutsu_name = response.css('h1.page-header__title::text').extract()[0]
        jutsu_name = jutsu_name.strip()

        yield dict(
            jutsu_name=jutsu_name
        )
