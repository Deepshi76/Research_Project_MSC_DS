import scrapy
import os
import json

class UstoreSpider(scrapy.Spider):
    name = "ustore"
    start_urls = [
        "https://www.ustore.lk/collections/axe",
        "https://www.ustore.lk/collections/ayush",
        # … all your other URLs …
    ]

    custom_settings = {
        "DOWNLOAD_DELAY": 1,
        # disable built‑in feed exporters
        "FEED_URI": None,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_items = []

    def parse(self, response):
        for card in response.css("div.product-grid-item"):
            detail_url = response.urljoin(card.css("a::attr(href)").get())
            item = {
                "name": card.css(".product-item__title::text").get().strip(),
                "price_now": card.css(".price-item--regular::text, .price-item--sale::text").get().strip(),
                "price_old": card.css(".price-item--compare::text").get(),
                "collection_url": response.url,
            }
            yield scrapy.Request(detail_url, callback=self.parse_product, meta={"item": item})

        next_page = response.css("link[rel=next]::attr(href)").get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)

    def parse_product(self, response):
        item = response.meta["item"]
        item.update({
            "product_url": response.url,
            "availability": response.css(".availability::text, .product-form__inventory::text").get(),
            "description": "\n".join(response.css("#ProductAccordion-product-template-description *::text, .rte *::text").getall()).strip(),
        })
        # compute discount_pct
        try:
            now = float(item["price_now"].replace("Rs.", "").replace(",", ""))
            old = float(item["price_old"].replace("Rs.", "").replace(",", "")) if item["price_old"] else now
            item["discount_pct"] = f"{round((old-now)/old*100)}%" if old and old > now else None
        except:
            item["discount_pct"] = None

        self.all_items.append(item)

    def closed(self, reason):
        # write out once the crawl is finished
        save_path = r"D:\Data_Science\Research\Deepshika_Rajendran_COMScDS232P-001_Research\Deepshika_Rajendran_COMScDS232P-001_Research_Code_File\Data\BrandINFO\ustore_full_dataset.txt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for item in self.all_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.logger.info(f"✅ Done! Data saved to: {save_path}")
